# Script to automatically generate the documentation pages for the plugins
import fuse
import os
import graphviz
import shutil

kind_colors = dict(
    geant4_interactions="#40C4F3",
    clustered_interactions="#FBCE56",
    tpc_interactions="#F44E43",
    below_cathode_interactions="#F44E43",
    interactions_in_roi="#56C46C",
    s1_photons="#54E4CF",
    s2_photons="#54E4CF",
    ap_photons="#54E4CF",
    propagated_photons="#54E4CF",
    pulse_ids="#54E4CF",
    pulse_windows="#F78C27",
    raw_records="#0260EF",
    individual_electrons="#F44E43",
)

# List of config options that are not tracked
config_options_not_tracked = [
    "debug",
    "raw_records_file_size_target",
    "min_records_gap_length_for_splitting",
    "input_file" "path",
    "file_name",
    "propagated_s2_photons_file_size_target",
    "min_electron_gap_length_for_splitting",
]

raw_html_text = """
.. raw:: html

{svg}
"""


def add_headline(headline, output, line):
    output += headline + "\n"
    output += line * len(headline) + "\n"
    return output


def add_tech_details(plugin, output):
    output += ".. code-block:: python\n\n"
    output += "   depends_on = " + str(plugin.depends_on) + "\n"
    output += "   provides = " + str(plugin.provides) + "\n"
    output += "   data_kind = " + str(plugin.data_kind) + "\n"
    output += "   __version__ = " + str(plugin.__version__) + "\n"

    if plugin.child_plugin:
        output += "\n"
        output += "   child_plugin = " + str(plugin.child_plugin) + "\n"
        output += "   parent_plugin_version = " + str(plugin.__class__.__base__.__version__) + "\n"

    return output


def add_columns_table(st, target, output):
    output += ".. list-table::\n"
    output += "   :widths: 25 25 50\n"
    output += "   :header-rows: 1\n\n"
    output += "   * - Field Name\n"
    output += "     - Data Type\n"
    output += "     - Comment\n"

    data_info_df = st.data_info(target)
    for i, row in data_info_df.iterrows():
        output += f"   * - {row['Field name']}" + "\n"
        output += f"     - {row['Data type']}" + "\n"
        output += f"     - {row['Comment']}" + "\n"

    return output


def get_config_df(st, target):
    config_df = st.show_config(target).sort_values(by="option")

    config_mask = []
    for ap_to in config_df["applies_to"].values:
        config_mask.append(any([target in a for a in ap_to]))
    keep_cols = ["option", "default", "help"]
    config_df = config_df[config_mask][keep_cols]
    return config_df


def add_config_table(st, target, output):
    output += ".. list-table::\n"
    output += "   :widths: 25 25 10 40\n"
    output += "   :header-rows: 1\n\n"
    output += "   * - Option\n"
    output += "     - default\n"
    output += "     - track\n"
    output += "     - Help\n"

    config_df = get_config_df(st, target)
    for i, row in config_df.iterrows():
        output += f"   * - {row['option']}" + "\n"
        output += f"     - {row['default']}" + "\n"

        if row["option"] in config_options_not_tracked:
            output += "     - False" + "\n"
        else:
            output += "     - True" + "\n"
        output += f"     - {row['help']}" + "\n"

    return output


def reformat_docstring(docstring):
    docstring = docstring.split("\n")

    for i, line in enumerate(docstring):
        if line.startswith("    "):
            docstring[i] = line[4:]

    y = "\n".join(docstring)
    return y


def add_deps_to_graph_tree(graph_tree, plugin, data_type, _seen=None):
    """Recursively add nodes to graph base on plugin.deps."""
    if _seen is None:
        _seen = []
    if data_type in _seen:
        return graph_tree, _seen

    # Add new one
    graph_tree.node(
        data_type,
        style="filled",
        href="#" + data_type.replace("_", "-"),
        fillcolor=kind_colors.get(plugin.data_kind_for(data_type), "grey"),
    )
    for dep in plugin.depends_on:
        graph_tree.edge(data_type, dep)

    # Add any of the lower plugins if we have to
    for lower_data_type, lower_plugin in plugin.deps.items():
        graph_tree, _seen = add_deps_to_graph_tree(graph_tree, lower_plugin, lower_data_type, _seen)
    _seen.append(data_type)
    return graph_tree, _seen


def add_spaces(x):
    """Add four spaces to every line in x.

    This is needed to make html raw blocks in rst format correctly
    """
    y = ""
    if isinstance(x, str):
        x = x.split("\n")
    for q in x:
        y += "    " + q
    return y


def create_plugin_documentation_text(st, plugin):

    output = ""

    plugin_name = plugin.__class__.__name__

    output += "=" * len(plugin_name) + "\n"
    output += plugin_name + "\n"
    output += "=" * len(plugin_name) + "\n"

    output += "\n"
    module = str(plugin.__module__).replace(".", "/")
    url_to_source = f"https://github.com/XENONnT/fuse/blob/master/{module}.py"
    output += f"Link to source: `{plugin.__class__.__name__} <{url_to_source}>`_ \n"
    output += "\n"

    output = add_headline("Plugin Description", output, "=")

    if plugin.__doc__:
        output += reformat_docstring(plugin.__doc__)
        output += "\n" * 2

    output = add_headline("Technical Details", output, "-")
    output += "\n"

    output = add_tech_details(plugin, output)
    output += "\n" * 2

    output = add_headline("Provided Columns", output, "=")
    output += "\n"

    targets = plugin.provides
    for target in targets:
        output = add_headline(target, output, "-")
        output += "\n"
        output = add_columns_table(st, target, output)
        output += "\n"

    output = add_headline("Config Options", output, "=")
    output += "\n"
    output = add_config_table(st, plugin.provides[0], output)
    output += "\n"

    output = add_headline("Dependency Graph", output, "=")
    output += "\n"

    graph_tree = graphviz.Digraph(format="svg")
    add_deps_to_graph_tree(graph_tree, plugin, target)
    fn = "." + "/graphs/" + target
    graph_tree.render(fn)
    with open(f"{fn}.svg", mode="r") as f:
        svg = add_spaces(f.readlines()[5:])

    output += raw_html_text.format(svg=svg)

    return output


def build_all_pages():

    st = fuse.context.full_chain_context(
        output_folder="./fuse_data", run_without_proper_corrections=True
    )

    all_registered_fuse_plugins = {}
    for key, value in st._plugin_class_registry.items():
        if "fuse" in str(value):
            all_registered_fuse_plugins[key] = value

    unique_plugins = {}
    for key, value in all_registered_fuse_plugins.items():
        unique_plugins[value] = key

    list_of_targets = list(unique_plugins.values())

    this_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plugins")

    for target in list_of_targets:
        plugin = st._get_plugins(targets=(target,), run_id="00000")[target]

        class_string = str(plugin.__class__)
        path_components = class_string.split(".")
        del path_components[-2]  # remove the file name
        file_path = os.path.join(this_dir, *path_components[2:])[:-2]

        documentation = create_plugin_documentation_text(st, plugin)

        file_name = file_path + ".rst"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(documentation)

    shutil.rmtree(os.path.dirname(os.path.realpath(__file__)) + "/graphs")


if __name__ == "__main__":
    build_all_pages()
