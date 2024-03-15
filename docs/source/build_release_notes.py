# Copied from straxen/docs/source/build_release_notes.py
# https://github.com/XENONnT/straxen/blob/a34a02393001f13755c3f84d15924f83bc86c4db
# /docs/source/build_release_notes.py

import os

from m2r import convert

header = """
Release notes
==============

"""


def convert_release_notes():
    """Convert the release notes to an RST page with links to PRs."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    notes = os.path.join(this_dir, "..", "..", "HISTORY.md")
    with open(notes, "r") as f:
        notes = f.read()
    rst = convert(notes)
    with_ref = ""
    for line in rst.split("\n"):
        # Get URL for PR
        if "#" in line:
            pr_number = line.split("#")[1]
            while len(pr_number):
                try:
                    pr_number = int(pr_number)
                    break
                except ValueError:
                    # Too many tailing characters to be an int
                    pr_number = pr_number[:-1]
            if pr_number:
                line = line.replace(
                    f"#{pr_number}",
                    f"`#{pr_number} <https://github.com/XENONnT/fuse/pull/{pr_number}>`_",
                )
        with_ref += line + "\n"
    target = os.path.join(this_dir, "release_notes.rst")

    with open(target, "w") as f:
        f.write(header + with_ref)


if __name__ == "__main__":
    convert_release_notes()
