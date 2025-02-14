import copy

import openmdao.core.group

"""Code for generating model definition using N2 viewer framework."""
from operator import itemgetter

import networkx as nx

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem, _SetupStatus
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.constants import _UNDEFINED
from openmdao.components.exec_comp import ExecComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.recorders.case_reader import CaseReader

# from openmdao.utils.array_utils import convert_ndarray_to_support_nans_in_json
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import default_noraise
from openmdao.utils.om_warnings import issue_warning
from openmdao import __version__ as openmdao_version

# from typing import Union

import json
import numpy as np
import json_numpy

json_numpy.patch()


def convert_nans_in_nested_list(val_as_list: list) -> None:
    """
    Given a list, possibly nested, replace any numpy.nan values with the string "nan".

    This is done since JSON does not handle nan. This code is used to pass variable values
    to the N2 diagram.

    The modifications to the list values are done in-place to avoid excessive copying of lists.

    Parameters
    ----------
    val_as_list : list
        List, possibly nested, whose nan elements need to be converted.
    """
    for i, val in enumerate(val_as_list):
        if isinstance(val, list):
            convert_nans_in_nested_list(val)
        else:
            if np.isnan(val):
                val_as_list[i] = "nan"
            elif np.isinf(val):
                val_as_list[i] = "infinity"
            else:
                val_as_list[i] = val


def convert_ndarray_to_support_nans_in_json(val: np.array) -> list:
    """
    Given numpy array of arbitrary dimensions, return the equivalent nested list with nan replaced.

    numpy.nan values are replaced with the string "nan".

    Parameters
    ----------
    val : ndarray
        Numpy array to be converted.

    Returns
    -------
    list
        The equivalent list (possibly nested) with any nan values replaced with the string "nan".
    """
    val = np.asarray(val)

    # do a quick check for any nans or infs and if not we can avoid the slow check
    nans = np.where(np.isnan(val))
    infs = np.where(np.isinf(val))
    if nans[0].size == 0 and infs[0].size == 0:
        return val.tolist()

    val_as_list = val.tolist()
    convert_nans_in_nested_list(val_as_list)
    return val_as_list


def _get_array_info(
    system: openmdao.core.group.System,
    vec: str,
    name: str,
    prom: str,
    var_dict: dict,
    from_src: bool = True,
) -> None:
    """

    Parameters
    ----------
    system : System
        The System at the root of the hierarchy
    vec: str
        Absolute name in the owning system's namespace.
    name: String
        Name of the value to get
    prom: str
        System component individual input or output
    var_dict: dict
        Dict of variables of prom
    from_src: bool
        If True, retrieve value of an input variable from its connected source.


    Returns
    -------
    """
    # if 'vec' is not None at this point, we can retrieve the value using vec._abs_get_val,
    # which is a faster call than system.get_val.
    ndarray_to_convert = (
        vec._abs_get_val(name, flat=False)
        if vec
        else system.get_val(prom, from_src=from_src)
    )

    var_dict["val"] = convert_ndarray_to_support_nans_in_json(ndarray_to_convert)

    # Find the minimum indices and value
    min_indices = np.unravel_index(
        np.nanargmin(ndarray_to_convert, axis=None), ndarray_to_convert.shape
    )
    var_dict["val_min_indices"] = min_indices
    var_dict["val_min"] = ndarray_to_convert[min_indices]

    # Find the maximum indices and value
    max_indices = np.unravel_index(
        np.nanargmax(ndarray_to_convert, axis=None), ndarray_to_convert.shape
    )
    var_dict["val_max_indices"] = max_indices
    var_dict["val_max"] = ndarray_to_convert[max_indices]


def _get_var_dict(
    system: openmdao.core.group.System,
    typ: str,
    name: str,
    is_implicit: bool,
    values: bool,
) -> dict:
    """
    Create dict from system model

    Parameters:
    ----------
    system: openmdao.core.group.System
        An openmdao System
    typ: str
        Variable type "input" or "output".
    name: str
        Node name
    is_implicit: bool
        Indicates if this is an implicit variable
    values: bool
        Indicates if values exist

    Returns:
    ----------
    var_dict: dict
        Variable dictionary from openmdao System
    """
    if name in system._var_abs2meta[typ]:
        meta = system._var_abs2meta[typ][name]
        prom = system._var_abs2prom[typ][name]
        val = np.asarray(meta["val"])
        is_dist = meta["distributed"]

        var_dict = {
            "name": prom,
            "type": typ,
            "dtype": type(val).__name__,
            "is_discrete": False,
            "distributed": is_dist,
            "shape": str(meta["shape"]),
            "desc": meta["desc"],
        }

        if typ == "output":
            var_dict["implicit"] = is_implicit
            vec = system._outputs
        else:  # input
            vec = None

        # if 'vec' is not None at this point, we can retrieve the value using vec._abs_get_val,
        # which is a faster call than system.get_val.

        if meta["units"] is None:
            var_dict["units"] = "None"
        else:
            var_dict["units"] = meta["units"]

        try:
            if values:
                # Get the current value
                _get_array_info(system, vec, name, prom, var_dict, from_src=True)
        except Exception as err:
            issue_warning(str(err))
    else:  # discrete
        meta = system._var_discrete[typ][name]
        val = meta["val"]
        var_dict = {
            "name": name,
            "type": typ,
            "dtype": type(val).__name__,
            "is_discrete": True,
        }
        if values:
            if isinstance(val, (int, str, list, dict, complex, np.ndarray)):
                var_dict["val"] = default_noraise(system.get_val(name))

    if "surrogate_name" in meta:
        var_dict["surrogate_name"] = meta["surrogate_name"]

    return var_dict


def _serialize_single_option(option: dict) -> object:
    #  -> Union[str, int, float, bool, list, tuple] :
    # tested
    """
    Return a json-safe equivalent of the option.

    The default_noraise function performs the datatype serialization, while this function takes
    care of attributes specific to options dicts.

    Parameters
    ----------
    option : object
        Option to be serialized.

    Returns
    -------
    object
       JSON-safe serialized object.
    """
    opt_keys = option.keys()
    if "recordable" not in opt_keys:
        return "Not Recordable"

    val = option["val"]

    if val is _UNDEFINED:
        return str(val)

    return default_noraise(val)


def _get_tree_dict(
    system: openmdao.core.group.System, values: bool = True, is_parallel: bool = False
) -> dict:
    """
    Get a dictionary representation of the system hierarchy.

    Parameters
    ----------
    system : System
        The System at the root of the hierarchy
    values : bool
        If True, include variable values. If False, all values will be None.
    is_parallel : bool
        If True, values can be remote and are not available.

    Returns
    ----------
    tree_dict: dict
        Returns tree dictionary of model
    """
    tree_dict = {
        "name": system.name if system.name else "root",
        "type": "subsystem" if system.name else "root",
        "class": ":".join((type(system).__module__, type(system).__qualname__)),
        "expressions": None,
        "nonlinear_solver": "",
        "nonlinear_solver_options": None,
        "linear_solver": "",
        "linear_solver_options": None,
    }
    is_implicit = False

    tree_dict["subsystem_type"] = "component"
    tree_dict["is_parallel"] = is_parallel
    if isinstance(system, ImplicitComponent):
        is_implicit = True
        tree_dict["component_type"] = "implicit"
        if overrides_method("solve_linear", system, ImplicitComponent):
            tree_dict["linear_solver"] = "solve_linear"
        elif system.linear_solver:
            tree_dict["linear_solver"] = system.linear_solver.SOLVER
            tree_dict["linear_solver_options"] = {
                k: _serialize_single_option(opt)
                for k, opt in system.linear_solver.options._dict.items()
            }

        if overrides_method("solve_nonlinear", system, ImplicitComponent):
            tree_dict["nonlinear_solver"] = "solve_nonlinear"
        elif system.nonlinear_solver:
            tree_dict["nonlinear_solver"] = system.nonlinear_solver.SOLVER
            tree_dict["nonlinear_solver_options"] = {
                k: _serialize_single_option(opt)
                for k, opt in system.nonlinear_solver.options._dict.items()
            }
    elif isinstance(system, ExecComp):
        tree_dict["component_type"] = "exec"
        tree_dict["expressions"] = system._exprs
    elif isinstance(system, (MetaModelStructuredComp, MetaModelUnStructuredComp)):
        tree_dict["component_type"] = "metamodel"
    elif isinstance(system, IndepVarComp):
        tree_dict["component_type"] = "indep"
    elif isinstance(system, ExplicitComponent):
        tree_dict["component_type"] = "explicit"
    else:
        tree_dict["component_type"] = None

    children = []
    for typ in ["input", "output"]:
        for abs_name in system._var_abs2meta[typ]:
            children.append(_get_var_dict(system, typ, abs_name, is_implicit, values))

        for prom_name in system._var_discrete[typ]:
            children.append(_get_var_dict(system, typ, prom_name, is_implicit, values))

    tree_dict["children"] = children

    options = {}
    slv = {"linear_solver", "nonlinear_solver"}
    for k, opt in system.options._dict.items():
        if k in slv:
            # need to handle solver option separately because it can be a class, instance or None
            try:
                val = opt["val"]
            except KeyError:
                val = opt["value"]

            try:
                options[k] = val.SOLVER
            except AttributeError:
                options[k] = val
        else:
            options[k] = _serialize_single_option(opt)

    tree_dict["options"] = options

    return tree_dict


def _get_declare_partials(system: openmdao.core.group.System) -> list:
    # tested
    """
    Get a list of the declared partials.

    Parameters
    ----------
    system : <System>
        A System in the model.

    Returns
    -------
    list
        A list containing all the declared partials (strings in the form "of > wrt" )
        beginning from the given system on down.
    """
    declare_partials_list = []
    for of, wrt in system._declared_partials_iter():
        if of != wrt:
            declare_partials_list.append(f"{of} > {wrt}")

    return declare_partials_list


def _get_viewer_data(
    data_source: openmdao.core.problem.Problem,
    values: bool = _UNDEFINED,
    case_id: int or None = None,
):
    """
    Get the data needed by the N2 viewer as a dictionary.

    Parameters
    ----------
    data_source : <Problem> or <Group> or str
        A Problem or Group or case recorder filename containing the model or model data.
        If the case recorder file from a parallel run has separate metadata, the
        filenames can be specified with a comma, e.g.: case.sql_0,case.sql_meta
    values : bool or _UNDEFINED
        If True, include variable values. If False, all values will be None.
        If unspecified, this behaves as if set to True unless the data source is a Problem or
        model for which setup is not complete, in which case it behaves as if set to False.
    case_id : int or str or None
        Case name or index of case in SQL file.

    Returns
    -------
    dict
        A dictionary containing information about the model for use by the viewer.
    """
    if isinstance(data_source, Problem):
        root_group = data_source.model

        if not isinstance(root_group, Group):
            # this function only makes sense when the model is a Group
            msg = (
                f"The model is of type {root_group.__class__.__name__}, "
                "viewer data is only available if the model is a Group."
            )
            raise TypeError(msg)

        driver = data_source.driver
        driver_name = driver.__class__.__name__
        driver_type = "doe" if isinstance(driver, DOEDriver) else "optimization"

        driver_options = {
            key: _serialize_single_option(driver.options._dict[key])
            for key in driver.options
        }

        if driver_type == "optimization" and hasattr(driver, "opt_settings"):
            driver_opt_settings = driver.opt_settings
        else:
            driver_opt_settings = None

        # set default behavior for values flag
        if values is _UNDEFINED:
            values = (
                data_source._metadata is not None
                and data_source._metadata["setup_status"]
                >= _SetupStatus.POST_FINAL_SETUP
            )

    elif isinstance(data_source, Group):
        if not data_source.pathname:  # root group
            root_group = data_source
            driver_name = None
            driver_type = None
            driver_options = None
            driver_opt_settings = None
        else:
            # this function only makes sense when it is at the root
            msg = (
                f"Viewer data is not available for sub-Group '{data_source.pathname}'."
            )
            raise TypeError(msg)

        # set default behavior for values flag
        if values is _UNDEFINED:
            values = (
                data_source._problem_meta is not None
                and data_source._problem_meta["setup_status"]
                >= _SetupStatus.POST_FINAL_SETUP
            )

    elif isinstance(data_source, str):
        if "," in data_source:
            filenames = data_source.split(",")
            cr = CaseReader(filenames[0], metadata_filename=filenames[1])
        else:
            cr = CaseReader(data_source)

        data_dict = cr.problem_metadata

        # set default behavior for values flag
        if values is _UNDEFINED:
            values = True

        def set_values(children, stack, case_):
            """
            Set variable values in model tree from the specified Case.

            If case is None, set all values to None.
            """
            for child in children:
                # if 'val' in child
                if child["type"] == "subsystem":
                    stack.append(child["name"])
                    set_values(child["children"], stack, case_)
                    stack.pop()
                elif child["type"] == "input":
                    if case_ is None:
                        child.pop("val")
                        for key in [
                            "val_min",
                            "val_max",
                            "val_min_indices",
                            "val_max_indices",
                        ]:
                            del child[key]
                    elif case_.inputs is None:
                        child["val"] = "N/A"
                    else:
                        path = (
                            child["name"]
                            if not stack
                            else ".".join(stack + [child["name"]])
                        )
                        child["val"] = case_.inputs[path]
                elif child["type"] == "output":
                    if case_ is None:
                        child.pop("val")
                        for key in [
                            "val_min",
                            "val_max",
                            "val_min_indices",
                            "val_max_indices",
                        ]:
                            del child[key]
                    elif case_.outputs is None:
                        child["val"] = "N/A"
                    else:
                        path = (
                            child["name"]
                            if not stack
                            else ".".join(stack + [child["name"]])
                        )
                        try:
                            child["val"] = case_.outputs[path]
                        except KeyError:
                            child["val"] = "N/A"

        if values is False:
            set_values(data_dict["tree"]["children"], [], None)
        elif case_id is not None:
            case = cr.get_case(case_id)
            print(f"Using source: {case.source}\nCase: {case.name}")
            set_values(data_dict["tree"]["children"], [], case)

        # Delete the variables key since it's not used in N2
        if "variables" in data_dict:
            del data_dict["variables"]

        # Older recordings might not have this.
        if "md5_hash" not in data_dict:
            data_dict["md5_hash"] = None

        return data_dict

    else:
        raise TypeError(
            f"Viewer data is not available for '{data_source}'."
            "The source must be a Problem, model or the filename of a recording."
        )

    data_dict = {
        "tree": _get_tree_dict(root_group, values=values),
        "md5_hash": root_group._generate_md5_hash(),
    }

    connections_list = []

    grp = root_group.compute_sys_graph(comps_only=True)

    scc = nx.strongly_connected_components(grp)

    strongdict = {}
    sys_idx_names = []

    for i, strong_comp in enumerate(scc):
        for c in strong_comp:
            strongdict[c] = i  # associate each comp with a strongly connected component

        if len(strong_comp) > 1:
            # these IDs are only used when back edges are present
            for name in strong_comp:
                sys_idx_names.append(name)

    sys_idx = (
        {}
    )  # map of pathnames to index of pathname in list (systems in cycles only)

    comp_orders = {
        name: i for i, name in enumerate(root_group._ordered_comp_name_iter())
    }
    for name in sorted(sys_idx_names):
        sys_idx[name] = len(sys_idx)

    # 1 is added to the indices of all edges in the matrix so that we can use 0 entries to
    # indicate that there is no connection.
    matrix = np.zeros((len(comp_orders), len(comp_orders)), dtype=np.int32)
    edge_ids = []
    for i, edge in enumerate(grp.edges()):
        src, tgt = edge
        if strongdict[src] == strongdict[tgt]:
            matrix[comp_orders[src], comp_orders[tgt]] = i + 1  # bump edge index by 1
            edge_ids.append((sys_idx[src], sys_idx[tgt]))
        else:
            edge_ids.append(None)

    for edge_i, (src, tgt) in enumerate(grp.edges()):
        if strongdict[src] == strongdict[tgt]:
            start = comp_orders[src]
            end = comp_orders[tgt]
            # get a view here, so we can remove this edge from submat temporarily to eliminate
            # an 'if' check inside the nested list comprehension for edges_list
            rem = matrix[start : start + 1, end : end + 1]
            rem[0, 0] = 0

            if end < start:
                start, end = end, start

            submat = matrix[start : end + 1, start : end + 1]
            nz = submat[submat > 0]

            rem[0, 0] = edge_i + 1  # put removed edge back

            if nz.size > 1:
                nz -= 1  # convert back to correct edge index
                edges_list = [edge_ids[i] for i in nz]
                edges_list = sorted(edges_list, key=itemgetter(0, 1))

                for vsrc, vtgtlist in grp.get_edge_data(src, tgt)["conns"].items():
                    for vtgt in vtgtlist:
                        connections_list.append(
                            {"src": vsrc, "tgt": vtgt, "cycle_arrows": edges_list}
                        )
                continue

        for vsrc, vtgtlist in grp.get_edge_data(src, tgt)["conns"].items():
            for vtgt in vtgtlist:
                connections_list.append({"src": vsrc, "tgt": vtgt})

    connections_list = sorted(connections_list, key=itemgetter("src", "tgt"))

    data_dict["sys_pathnames_list"] = list(sys_idx)
    data_dict["connections_list"] = connections_list
    data_dict["abs2prom"] = root_group._var_abs2prom

    data_dict["driver"] = {
        "name": driver_name,
        "type": driver_type,
        "options": driver_options,
        "opt_settings": driver_opt_settings,
    }
    data_dict["design_vars"] = root_group.get_design_vars(use_prom_ivc=False)
    data_dict["responses"] = root_group.get_responses(use_prom_ivc=False)

    data_dict["declare_partials_list"] = _get_declare_partials(root_group)

    return data_dict


def model_to_dict(
    data_source: openmdao.core.problem.Problem,
    path: str or None = None,
    values: bool or _UNDEFINED = _UNDEFINED,
    case_id: str or None = None,
    title: str or None = None,
) -> dict:
    """
    Generate an JSON file containing the model.
    Pulled from N2 Viewer code

    Parameters
    ----------
    data_source : <Problem> or str
        The Problem or case recorder database containing the model or model data.
    path : str, optional
        If specified, the n2 viewer will begin in a state that is zoomed in on the selected path.
        This path should be the absolute path of a system in the model.
    values : bool or _UNDEFINED
        If True, include variable values. If False, all values will be None.
        If unspecified, this behaves as if set to True unless the data source is a Problem or
        model for which setup is not complete, in which case it behaves as if set to False.
    case_id : int, str, or None
        Case name or index of case in SQL file if data_source is a database.
    title : str, optional
        The title for the diagram. Used in the HTML title.

    Returns
    ----------
    json_vars: dict
        Model represented as dict
    """
    # grab the model viewer data
    model_data = _get_viewer_data(data_source, values=values, case_id=case_id)

    options = {}
    model_data["options"] = options

    if title:
        title = f"OpenMDAO Model Hierarchy and N2 diagram: {title}"
    else:
        title = "OpenMDAO Model Hierarchy and N2 diagram"

    json_vars = {
        "title": title,
        "openmdao_version": openmdao_version,
        "model_data": model_data,
        "initial_path": path,
    }

    return json_vars


def write_json(json_vars_: dict, outfile: str) -> None:
    """
    Write model dictionary json_vars_ to file outfile.
    Parameters
    ----------
     json_vars_:  dict
        Dictionary of model description

     param outfile: str
        Fully qualified path of output data file
    Returns
    -------
     Returns nothing

    """
    success_flag = False
    # replace filter_numpy with jason_numpy
    # json_vars_new = filter_numpy(json_vars_)
    json_vars_new = json_vars_
    with open(outfile, "w") as out_dict_f:
        # change numpy values to plain
        out_dict = json.dumps(json_vars_new)
        out_dict_json = json.loads(out_dict)
        out_dict_f.write(json.dumps(out_dict_json, indent=4, sort_keys=True))


def pull_connections_list(full_model_dict: dict) -> list:
    # tested
    """
    Returns a list of the components which make up the system model.

    Parameters
    ----------
    full_model_dict: dict
        Dict of system model

    Returns
    ----------
    circuit_components: list
        List of the components which make up the system model.
    """
    # pull out connections list
    connections_list = full_model_dict["model_data"]["connections_list"]
    # pull system pathnames list to decode connection list cycle arrows
    sys_pathnames_list = full_model_dict["model_data"]["sys_pathnames_list"]

    for d in connections_list:
        if "cycle_arrows" in d:
            temp_list = []
            for prs in d["cycle_arrows"]:
                nams = [
                    sys_pathnames_list[int(prs[0])],
                    sys_pathnames_list[int(prs[1])],
                ]
                temp_list.append(nams)
            d["cycle_names"] = temp_list

    # get all systems
    circuit = full_model_dict["model_data"]["tree"]["children"]
    # get base name
    base_name = list(full_model_dict["model_data"]["responses"].keys())[0].split(".")[0]
    # get component names
    circuit_components = []
    for cir in circuit:
        # if cir["name"].startswith("circuit"):
        if cir["name"].startswith(base_name):
            cir_tok = cir["name"].split(".")
            circuit_components.append(cir_tok[1])
    circuit_components = list(set(circuit_components))

    print("\n")
    for key in circuit_components:
        print(f"Find info in some other file for {key}")
    return circuit_components


def get_additional_options(model_dict: dict, add_option_file_: str) -> dict:
    """
    Reads file containing additional parameters to be added to madel
    TODO: This function needs to be replaced with another that programmatically retrieves the additional parameters.

    Parameters
    ----------
    model_dict: dict
        OpenMDAO system model.

    add_option_file_: str
        Path to file containing a dict of additional options to add to model.

    Returns
    ----------
    """
    # tested
    with open(add_option_file_, "r") as optF:
        additional_options_dict = json.load(optF)
        options_keys = additional_options_dict.keys()
        temp_opt_dict = {}
        for comp in options_keys:
            temp_comp_dict = {}
            for comp_keys in additional_options_dict[comp]:
                temp_comp_dict[comp_keys] = additional_options_dict[comp][comp_keys]
            temp_opt_dict[comp] = temp_comp_dict
        # Add options to model dict
        model_dict["Optional"] = temp_opt_dict
        return model_dict


def get_cycle_names(temp_dict: dict) -> dict:
    """
    Converts cycle_arrow numbers to component connection names
     Parameters
    ----------
     temp_dict:  dict
        Dictionary of model description

    Returns
    -------
     out_dict: dict
        Original dictionary adding cycle_names converted form cycle_arrow numbers
    """
    out_dict = copy.deepcopy(temp_dict)
    # pull out connections list
    connections_list = out_dict["model_data"]["connections_list"]
    # print(f'connections_list:\n{connections_list}')
    ###############################################
    # pull system pathnames list to decode connection list cycle arrows
    sys_pathnames_list = out_dict["model_data"]["sys_pathnames_list"]
    print(f"sys_pathnames_list:\n{sys_pathnames_list}")

    for d in connections_list:
        if "cycle_arrows" in d:
            temp_list = []
            for prs in d["cycle_arrows"]:
                nams = [
                    sys_pathnames_list[int(prs[0])],
                    sys_pathnames_list[int(prs[1])],
                ]
                temp_list.append(nams)
            seen_list = set()
            single_list = [
                x
                for x in temp_list
                if tuple(x[::-1]) not in seen_list and not seen_list.add(tuple(x))
            ]
            d["cycle_names"] = temp_list
            d["cycle_names_single"] = single_list
    return out_dict
