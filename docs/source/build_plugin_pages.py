# Script to automatically generate the documentation pages for the plugins

import fuse
import os


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
        # Can i get this somehow...?
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
        file_path = os.path.join(
            this_dir, class_string.split(".")[2], *class_string.split(".")[4:]
        )[:-2]

        documentation = create_plugin_documentation_text(st, plugin)

        file_name = file_path + ".rst"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(documentation)


if __name__ == "__main__":
    build_all_pages()
