import warnings
import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache
from typing import Protocol, runtime_checkable
from copy import deepcopy
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.shear import PBJTATTLoopComputer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from scipy.interpolate import CubicSpline


def _nonlinear_bias_loop_computer(perturbations):
    """Default one-loop kernels for galaxy_bias_model="nonlinear", imported
    here so cloelike also works with cloelib versions without it."""
    try:
        from cloelib.observables.photo.positions import PBJNonlinearBiasLoopComputer
    except ImportError as e:
        raise ImportError(
            "galaxy_bias_model='nonlinear' needs a cloelib with "
            "PBJNonlinearBiasLoopComputer (cloelib PR #384)."
        ) from e
    return PBJNonlinearBiasLoopComputer(perturbations)


# Intrinsic-alignment models accepted in settings["ia_model"], mapped to the
# (required, optional) nuisance parameters ShearTracer reads for each. Optional
# ones fall back to cloelib's defaults when absent from `parameters`.
IA_MODEL_KEYS = {
    None: ((), ()),
    "NLA": (("AIA", "EtaIA", "CIA"), ()),
    "TATT": (("AIA", "A2IA", "bTA"), ("EtaIA", "Eta2IA", "CIA", "z0IA")),
}

# Galaxy bias models accepted in settings["galaxy_bias_model"].
GALAXY_BIAS_MODELS = ("poly", "per_bin", "per_bin_int", "nonlinear")


@runtime_checkable
class PhotoLikelihoodProtocol(Protocol):
    """
    Protocol for photo-z likelihood classes.

    This protocol defines the required interface for photometric likelihood implementations,
    specifying initialization, required attributes, and methods for computing data vectors,
    theory vectors, covariance matrices, and log-likelihoods.

    Attributes:
        data (dict): Observational data dictionary.
        settings (dict): Configuration settings dictionary.
        Background (Background): Cosmological background instance.
        LinPerturbations (Perturbations): Linear perturbations instance.
        NonLinPerturbations (Perturbations): Non-linear perturbations instance.
        derived (dict): Dictionary for derived quantities.
        mode (str): Mode of operation (e.g., "coupled").

    Methods:
        __init__(data, settings, Background, LinPerturbations, NonLinPerturbations, mode):
            Initializes the likelihood protocol.
        get_data_vector_full() -> np.ndarray:
            Returns the full data vector.
        get_data_vector_masked() -> np.ndarray:
            Returns the masked data vector.
        get_theory_vector_full(parameters: dict) -> np.ndarray:
            Returns the full theory vector for given parameters.
        get_theory_vector_masked(parameters: dict) -> np.ndarray:
            Returns the masked theory vector for given parameters.
        get_covariance_matrix_full() -> np.ndarray:
            Returns the full covariance matrix.
        get_covariance_matrix_masked_inv() -> np.ndarray:
            Returns the inverse of the masked covariance matrix.
        loglike(parameters: dict) -> float:
            Computes the log-likelihood for the given parameters.
    """

    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: Background,
        LinPerturbations: Perturbations,
        NonLinPerturbations: Perturbations,
    ) -> None: ...

    data: dict
    settings: dict
    Background: Background
    LinPerturbations: Perturbations
    NonLinPerturbations: Perturbations
    derived: dict

    def get_data_vector_full(self) -> np.ndarray: ...
    def get_data_vector_masked(self) -> np.ndarray: ...
    def get_theory_vector_full(self, parameters: dict) -> np.ndarray: ...
    def get_theory_vector_masked(self, parameters: dict) -> np.ndarray: ...
    def get_covariance_matrix_full(self) -> np.ndarray: ...
    def get_covariance_matrix_masked_inv(self) -> np.ndarray: ...
    def loglike(self, parameters: dict) -> float: ...


class PhotoLikelihoodBase:
    """
    Base class for photometric likelihood calculations using angular correlation functions.
    This class provides methods for preparing, binning, and masking data, as well as computing
    likelihoods based on theoretical predictions and observed data vectors.
    Args:
        data (dict): Dictionary containing observational data, including '2pcf', 'theta', 'z_arr', and 'cov'.
        settings (dict): Configuration settings. Besides the required 'scale_cuts',
            the theory model is selected with the optional keys:

            - 'ia_model': None, "NLA" (default) or "TATT". NLA reads AIA, EtaIA, CIA;
              TATT reads AIA, A2IA, bTA and optionally EtaIA, Eta2IA, CIA, z0IA.
            - 'galaxy_bias_model': "poly" (default, b1_photo_poly0..3), "per_bin" or
              "per_bin_int" (b1_photo_bin{i}, i = 0..n_pos_bins-1), or "nonlinear"
              (b1_photo_nl_bin{i}, and optionally b2_, bs2_, b3nl_, bk2_photo_nl_bin{i}).
            - 'include_rsd': bool, default False.
            - 'nonlinear_param_keys': names of sampled parameters passed as keyword
              arguments to NonLinPerturbations, default ("log10TAGN",). Use () for
              backends without baryonic feedback (e.g. EE2).
            - 'baryon_param_keys': names of sampled parameters passed as
              `baryon_kwargs` to NonLinPerturbations, for classes built with
              `cloelib.cosmology.cosmology.with_baryon_boost`. Default ().
            - 'nonlinear_kwargs': fixed keyword arguments for NonLinPerturbations
              (e.g. {"nonlinear_model": "mead2020"}). Default {}.
            - 'coupled_ell_nodes': in coupled mode, compute C_l on sparse multipole nodes
              (every l from 2 to ``dense_below``, then ``per_decade`` per decade to the
              top of the mixing matrices) and carry them to every l by a cubic spline of
              l (l + 1) C_l in ln l, folded into each mixing matrix once; e.g.
              {"dense_below": 50, "per_decade": 40}. Default None: every l.
            - 'tatt_loop_computer', 'nl_bias_loop_computer': factories called with
              the linear perturbations to build the one-loop kernels for TATT and
              the nonlinear galaxy bias. Default to cloelib's FAST-PT computers.
        Background: Object representing background cosmology.
        LinPerturbations: Object representing linear perturbations.
        NonLinPerturbations: Object representing non-linear perturbations.
        mode (str, optional): Mode of operation, default is "coupled".
    Attributes:
        data (dict): Observational data.
        settings (dict): Configuration settings.
        derived (dict): Dictionary for storing derived quantities.
        Background: Background cosmology object.
        LinPerturbations: Linear perturbations object.
        NonLinPerturbations: Non-linear perturbations object.
        scale_cuts: Scale cuts from settings.
        zs: Redshift array from data.
        mixmat: Mixing matrix, possibly rebinned.
        weight_mat: Weight matrix used for binning.
        masking_vector: Boolean mask for selecting data vector elements.
    Methods:
        _masking(arr, interval):
            Returns a boolean mask for elements within the specified interval.
        get_covariance_matrix_full():
            Returns the full covariance matrix from the data.
        get_data_vector_masked():
            Returns the masked data vector.
        get_covariance_matrix_masked_inv():
            Returns the inverse of the masked covariance matrix.
        get_theory_vector_full(parameters):
            Returns the full theoretical prediction vector for given parameters.
        get_theory_vector_masked(parameters):
            Returns the masked theoretical prediction vector for given parameters.
        loglike(parameters):
            Computes the log-likelihood for the given parameters using the masked data and theory vectors.
    """

    def __init__(
        self,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        ells_integration=None,
        mode="coupled",
    ):
        self.data = data
        self.settings = settings
        self.derived = {}
        self.Background = Background
        self.LinPerturbations = LinPerturbations
        self.NonLinPerturbations = NonLinPerturbations
        self.theory_prediction = {}
        self.mode = mode
        if "cosebis" in data:
            # For COSEBIs the scale cuts are set by the W_ell kernels
            if settings.get("scale_cuts") is not None:
                warnings.warn(
                    "For COSEBIs the scale cuts are applied via the w_ells; "
                    "settings['scale_cuts'] is ignored."
                )
            self.scale_cuts = None
        else:
            self.scale_cuts = settings["scale_cuts"]
        self.ia_model = settings.get("ia_model", "NLA")
        if self.ia_model not in IA_MODEL_KEYS:
            raise ValueError(
                f"Unknown ia_model {self.ia_model!r}; "
                f"choose one of {list(IA_MODEL_KEYS)}."
            )
        self.galaxy_bias_model = settings.get("galaxy_bias_model", "poly")
        if self.galaxy_bias_model not in GALAXY_BIAS_MODELS:
            raise ValueError(
                f"Unknown galaxy_bias_model {self.galaxy_bias_model!r}; "
                f"choose one of {list(GALAXY_BIAS_MODELS)}."
            )
        if self.galaxy_bias_model == "nonlinear" and settings.get("include_rsd", False):
            # Fail here rather than at the first loglike call.
            raise ValueError(
                "galaxy_bias_model='nonlinear' does not support include_rsd=True "
                "yet: cloelib's generalized Cl engine has no RSD term."
            )
        self.nonlinear_param_keys = tuple(
            settings.get("nonlinear_param_keys", ("log10TAGN",))
        )
        self.baryon_param_keys = tuple(settings.get("baryon_param_keys", ()))
        self.rebin = False
        self.zs = data["z_arr"]
        # Redshifts of the matter power spectrum: settings["pk_redshifts"] uniform nodes from
        # 0 to the top of z_arr if given (CAMB takes at most 256), else z_arr itself.
        n_pk = settings.get("pk_redshifts")
        self.pk_zs = self.zs if n_pk is None else np.linspace(0.0, np.max(self.zs), int(n_pk))
        if self.mode == "coupled":
            self.mixmat = deepcopy(data.get("mixmat", {}))
        else:
            self.mixmat = None
        self._coupling = None

        if (ells_integration is None) and ("2pcf" in data):
            self.ells_integration = np.arange(2, 40000)

        else:
            self.ells_integration = ells_integration

        if ("cells" in data) and ("n_ell_bins" in settings):
            self._prepare()

    @staticmethod
    def sparse_ells(lmax, dense_below, per_decade):
        """Multipole nodes: every l in ``2..dense_below``, then log-spaced to ``lmax``."""
        if lmax <= dense_below:
            return np.arange(2, lmax + 1)
        n = max(int(np.ceil(per_decade * np.log10(lmax / dense_below))), 2) + 1
        tail = np.round(np.geomspace(dense_below, lmax, n)).astype(int)
        return np.unique(np.concatenate([np.arange(2, dense_below + 1), tail]))

    @staticmethod
    def interpolation_matrix(nodes, n_ell):
        """``(n_ell, len(nodes))``: C_l at ``nodes`` to every l < ``n_ell`` by a cubic spline
        of l (l + 1) C_l in ln l; the l = 0, 1 rows are zero."""
        matrix = np.zeros((n_ell, len(nodes)))
        ells = np.arange(2, n_ell)
        spline = CubicSpline(np.log(nodes), np.diag(nodes * (nodes + 1.0)), axis=0)
        matrix[2:] = spline(np.log(ells)) / (ells * (ells + 1.0))[:, None]
        return matrix

    def _build_coupling(self):
        """The multipoles C_l is computed at in coupled mode, and for each spectrum the
        operator from C_l there to its bandpowers: its mixing matrix (the EE<-EE component
        for shear x shear), times the spline to every l when ``coupled_ell_nodes`` is set."""
        matrices = {}
        for key, mixmat in self.mixmat.items():
            m = np.asarray(mixmat.array, dtype=float)
            matrices[key] = m[0] if key[:2] == ("SHE", "SHE") else m
        n_ell = max(m.shape[-1] for m in matrices.values())
        nodes = self.settings.get("coupled_ell_nodes")
        if nodes is None:
            self.coupled_ells = np.arange(n_ell)
            self._coupling = matrices
            return
        self.coupled_ells = self.sparse_ells(n_ell - 1, nodes["dense_below"], nodes["per_decade"])
        spline = self.interpolation_matrix(self.coupled_ells, n_ell)
        self._coupling = {key: m @ spline[: m.shape[-1]] for key, m in matrices.items()}

    def _theory_block(self, tracer1, tracer2, keys, nlp, component):
        """The theory of the spectra ``keys`` of one tracer pair, concatenated: in coupled
        mode their bandpowers, each spectrum's operator (`_build_coupling`) on C_l at the
        coupled multipoles; decoupled, ``component`` of each packaged C_l at the data's
        multipoles (kept in ``theory_prediction``)."""
        two_point = AngularTwoPoint(tracer1, tracer2)
        if self.mode == "coupled":
            if self._coupling is None:
                self._build_coupling()
            cl = two_point.get_Cl_tensor(self.coupled_ells, 0, nlp.k)
            return jnp.concatenate(
                [self._coupling[key] @ cl[:, key[2] - 1, key[3] - 1] for key in keys]
            )
        cells = two_point.get_Cl(self.data["ells"], 0, nlp.k)
        self.theory_prediction.update(cells)
        return jnp.concatenate([jnp.ravel(component(cells[key])) for key in keys])

    def _prepare(self):
        """Perform rebinning if required by settings."""
        if self.settings["n_ell_bins"] < len(self.data["ells"]):
            self.rebin = True
            self.data["cells_unbin"] = self.data["cells"]
            self.data["ells_unbin"] = self.data["ells"]
            self._bin_data(
                self.data["cells"], self.data["ells"], self.settings["n_ell_bins"]
            )
            self._bin_mixmat()

    def _bin_data(self, cells_data, ells, n_bins):
        """Geometric rebinning of data vectors."""
        bin_edges = np.geomspace(10, ells[-1], n_bins + 1)
        mask_bins = [
            (ells >= bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)
        ]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]
        self.data["ells"] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        for k in cells_data.keys():
            self.data["cells"][k] = cells_data[k] @ self.weight_mat.T

    def _bin_mixmat(self):
        """Apply the same binning to the mixing matrix."""
        for k in self.mixmat.keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
            if k[:2] == ("SHE", "SHE"):
                new_array = np.transpose(new_array, axes=(1, 0, 2))
            self.mixmat[k] = new_array

    def _masking(self, arr, interval):
        return (arr >= interval[0]) & (arr <= interval[1])

    # Parameters passed to Background(...). Parameters consumed only by
    # NonLinPerturbations (e.g. log10TAGN) are set per instance through
    # settings["nonlinear_param_keys"] and settings["baryon_param_keys"].
    _BACKGROUND_PARAM_KEYS = (
        "H0",
        "Omega_cdm0",
        "Omega_b0",
        "Omega_k0",
        "w0",
        "wa",
        "ns",
        "As",
        "mnu",
        "gamma_MG",
        "N_mnu",
        "alpha_s",
    )

    def _init_she_keys(self):
        """Shear nuisance parameters required by the selected IA model."""
        n = self.data["dndz_she"].shape[0]
        ia_required, ia_optional = IA_MODEL_KEYS[self.ia_model]
        self.full_she_keys = (
            list(ia_required)
            + [f"multiplicative_bias_{i}" for i in range(1, n + 1)]
            + [f"dz_shear_{i}" for i in range(1, n + 1)]
            + [f"width_shear_{i}" for i in range(1, n + 1)]
        )
        self.optional_she_keys = list(ia_optional)

    def _init_pos_keys(self):
        """Position nuisance parameters required by the selected bias model."""
        n = self.data["dndz_pos"].shape[0]
        if self.galaxy_bias_model == "poly":
            bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
            optional = []
        elif self.galaxy_bias_model == "nonlinear":
            # Higher-order terms are optional: cloelib drops a term whose
            # coefficient is absent.
            bias_keys = [f"b1_photo_nl_bin{i}" for i in range(n)]
            optional = [
                f"{p}_photo_nl_bin{i}"
                for p in ("b2", "bs2", "b3nl", "bk2")
                for i in range(n)
            ]
        else:
            bias_keys = [f"b1_photo_bin{i}" for i in range(n)]
            optional = []
        self.full_pos_keys = (
            bias_keys
            + [f"magnification_bias_{i}" for i in range(1, n + 1)]
            + [f"dz_pos_{i}" for i in range(1, n + 1)]
            + [f"width_pos_{i}" for i in range(1, n + 1)]
        )
        self.optional_pos_keys = optional

    @staticmethod
    def _select(parameters, required, optional=()):
        """Subset of `parameters`: all `required` keys, plus the `optional`
        ones that are present."""
        selected = {k: parameters[k] for k in required}
        selected.update({k: parameters[k] for k in optional if k in parameters})
        return selected

    def _memo(self, slot, parameters, values, build):
        """``build()``, reused while ``values`` are unchanged. When any of them is traced
        (the likelihood under ``jax.jit`` or ``jax.vmap``), values cannot be compared, and it
        is reused only for the same ``parameters`` dict, i.e. within one evaluation."""
        traced = any(isinstance(v, jax.core.Tracer) for v in values)
        cached = getattr(self, slot, None)
        if cached is not None:
            key, value = cached
            if key is parameters if traced else (isinstance(key, tuple) and key == values):
                return value
        value = build()
        setattr(self, slot, (parameters if traced else tuple(values), value))
        return value

    def _get_perturbations(self, parameters):
        """Build (and cache) Background/LinPerturbations/NonLinPerturbations for
        the given parameters.

        Combined likelihoods (e.g. 3x2pt) mix several probe-specific mixins in
        one MRO chain, each of which needs the perturbations. Caching here
        (instead of rebuilding in every mixin's get_theory_vector_full) avoids
        recomputing the same cosmology multiple times per call. The cache is
        keyed on the cosmology-relevant subset of `parameters` so a change in
        any of those values rebuilds it, while repeated calls with an unchanged
        cosmology (e.g. only nuisance parameters varying) reuse it.
        """
        return self._memo(
            "_perturbations_cache",
            parameters,
            self._cosmo_key(parameters),
            lambda: self._build_perturbations(parameters),
        )

    def _build_perturbations(self, parameters):
        background = self.Background(
            **{k: parameters[k] for k in self._BACKGROUND_PARAM_KEYS}
        )
        lp = self.LinPerturbations(background=background, redshifts=self.pk_zs)
        nl_kwargs = dict(self.settings.get("nonlinear_kwargs", {}))
        nl_kwargs.update({k: parameters[k] for k in self.nonlinear_param_keys})
        if self.baryon_param_keys:
            nl_kwargs["baryon_kwargs"] = {
                k: parameters[k] for k in self.baryon_param_keys
            }
        nlp = self.NonLinPerturbations(
            background=background,
            linearperturbations=lp,
            redshifts=self.pk_zs,
            **nl_kwargs,
        )
        # sigma8_0() self-memoizes by overwriting `nlp.sigma8_0` with its
        # result on first call, so it must be called exactly once per nlp
        # instance -- here, rather than in each mixin (which would fail on
        # the second mixin to see this shared, cached nlp).
        self.derived["sigma8_0"] = nlp.sigma8_0()
        # Loop computers depend only on the cosmology; drop the old ones.
        self._loop_computers = {}
        return background, lp, nlp

    def _cosmo_key(self, parameters):
        """The same cache key used by `_get_perturbations`, exposed so tracer
        caching below can be keyed on values rather than on `nlp`'s object
        identity (which could in principle be reused after garbage collection)."""
        keys = (
            self._BACKGROUND_PARAM_KEYS
            + self.nonlinear_param_keys
            + self.baryon_param_keys
        )
        return tuple(parameters[k] for k in keys)

    def _get_loop_computer(self, name, default_factory, parameters):
        """One-loop kernel computer for TATT ("tatt") or the nonlinear galaxy
        bias ("nl_bias"), built once per cosmology so the FAST-PT kernels are
        shared between tracers and nuisance-only steps.

        It is built from the linear perturbations: the one-loop integrals need
        the linear power spectrum, which not every nonlinear backend exposes.
        """
        _, lp, _ = self._get_perturbations(parameters)
        if name not in self._loop_computers:
            factory = self.settings.get(f"{name}_loop_computer", default_factory)
            self._loop_computers[name] = factory(lp)
        return self._loop_computers[name]

    def _get_pos_tracer(self, parameters, nlp):
        """Build (and cache) the PositionsTracer for the given parameters/nlp,
        so GCph and GGL do not each rebuild it within the same call."""
        nuisance = self._select(parameters, self.full_pos_keys, self.optional_pos_keys)
        values = self._cosmo_key(parameters) + tuple(v for _, v in sorted(nuisance.items()))
        return self._memo(
            "_pos_tracer_cache", parameters, values,
            lambda: self._build_pos_tracer(parameters, nlp, nuisance),
        )

    def _build_pos_tracer(self, parameters, nlp, nuisance):
        # The nonlinear bias options are only passed when selected, so the linear
        # bias models also work with cloelib versions without the nonlinear bias.
        nl_bias_kwargs = {}
        if self.galaxy_bias_model == "nonlinear":
            nl_bias_kwargs["nl_bias_loop_computer"] = self._get_loop_computer(
                "nl_bias", _nonlinear_bias_loop_computer, parameters
            )
        tracer = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params=nuisance,
            galaxy_bias_model=self.galaxy_bias_model,
            include_rsd=self.settings.get("include_rsd", False),
            **nl_bias_kwargs,
        )
        return tracer

    def _get_she_tracer(self, parameters, nlp):
        """Build (and cache) the ShearTracer for the given parameters/nlp, so
        WL and GGL do not each rebuild it within the same call."""
        nuisance = self._select(parameters, self.full_she_keys, self.optional_she_keys)
        values = self._cosmo_key(parameters) + tuple(v for _, v in sorted(nuisance.items()))
        return self._memo(
            "_she_tracer_cache", parameters, values,
            lambda: self._build_she_tracer(parameters, nlp, nuisance),
        )

    def _build_she_tracer(self, parameters, nlp, nuisance):
        if self.ia_model == "TATT":
            loop_computer = self._get_loop_computer(
                "tatt", PBJTATTLoopComputer, parameters
            )
        else:
            loop_computer = None
        tracer = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params=nuisance,
            ia_model=self.ia_model,
            tatt_loop_computer=loop_computer,
        )
        return tracer

    def get_masking_vector(self):
        return np.array([], dtype=bool)

    def get_covariance_matrix_full(self):
        return self.data["cov"]

    def get_data_vector_full(self):
        return np.array([])

    @lru_cache(maxsize=None)
    def get_masking_vector_cached(self):
        """Compute (once) and cache the combined boolean mask."""
        return self.get_masking_vector()

    @lru_cache(maxsize=None)
    def get_data_vector_masked(self):
        mask = self.get_masking_vector_cached()
        return self.get_data_vector_full()[mask]

    @lru_cache(maxsize=None)
    def get_covariance_matrix_masked_inv(self):
        """Compute (once) and cache inverse masked covariance matrix."""
        cov = self.get_covariance_matrix_full()
        mask = self.get_masking_vector_cached()
        cov_masked = cov[mask][:, mask]
        return np.linalg.inv(cov_masked)

    def get_theory_vector_full(self, parameters):
        return np.array([])

    def get_theory_vector_masked(self, parameters):
        mask = self.get_masking_vector_cached()
        return self.get_theory_vector_full(parameters)[mask]

    def loglike(self, parameters):
        diff = self.get_theory_vector_masked(parameters) - self.get_data_vector_masked()
        inv_cov = self.get_covariance_matrix_masked_inv()
        return -0.5 * diff @ inv_cov @ diff
