import glob
import os
import re
import setuptools

# ----- overrides -----
# set these to anything but None to override the automatic defaults
packages = None
package_name = None
package_data = None
scripts = None
requirements_file = None
requirements = None
dependency_links = None
version = "0.0"


def find_scripts():
    return [s for s in setuptools.findall("scripts/") if os.path.splitext(s)[1] != ".pyc"]


def package_to_path(package_path):
    """Convert a package (as found by setuptools.find_packages) e.g. "foo.bar" to "foo/bar"."""
    return package_path.replace(".", "/")


def find_subdirectories(package_path):
    """Get the subdirectories within a package. This will include resources (non-submodules) and submodules."""
    try:
        subdirectories = [f for f in glob.glob(package_to_path(package_path) + "**/", recursive=True)]
    except StopIteration:
        subdirectories = []
    return subdirectories


def subdir_findall(directory, subdir):
    """Find all files in a subdirectory and return paths relative to dir
    This is similar to (and uses) setuptools.findall However, the paths returned are in the form
    needed for package_data
    """
    strip_n = len(directory.split("/"))
    path = "/".join((directory, subdir))
    return ["/".join(s.split("/")[strip_n:]) for s in setuptools.findall(path)]


def find_package_data(packages_list):
    """For a list of packages, find the package_data.
    This function scans the subdirectories of a package and considers all non-submodule subdirectories as resources,
    including them in the package_data. Returns a dictionary suitable for setup(package_data=<result>)
    """
    data = {}
    for p in packages_list:
        data[p] = []
        for subdir in find_subdirectories(p):
            data[p] += subdir_findall(package_to_path(p), subdir)
    return data


def parse_requirements(file_name):
    """Parse package requirements from requirements.txt file.
    http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
    """
    reqs = []
    with open(file_name, "r") as f:
        for line in f:
            if re.match(r"(\s*#)|(\s*$)", line):
                continue
            if re.match(r"\s*-e\s+", line):
                reqs.append(re.sub(r"\s*-e\s+.*#egg=(.*)$", r"\1", line).strip())
            elif re.match(r"\s*-f\s+", line):
                pass
            else:
                reqs.append(line.strip())
    return reqs


def parse_dependency_links(file_name):
    """Parse package dependencies from requirements.txt file.
    http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
    """
    dep_links = []
    with open(file_name) as f:
        for line in f:
            if re.match(r"\s*-[ef]\s+", line):
                dep_links.append(re.sub(r"\s*-[ef]\s+", "", line))
    return dep_links


# Set logging mode to debug (if debug flag is set).
if not (hasattr(setuptools, "dist") and setuptools.dist):
    raise ImportError("distribute was not found and fallback to setuptools was not allowed")
# Examine directory, i.e. search for package name, data, scripts, etc.
if packages is None:
    packages = setuptools.find_packages()
if len(packages) == 0:
    raise Exception("No valid packages found")
if package_name is None:
    package_name = packages[0]
if package_data is None:
    package_data = find_package_data(packages)
if scripts is None:
    scripts = find_scripts()
if requirements_file is None:
    requirements_file = "requirements.txt"
if os.path.exists(requirements_file):
    if requirements is None:
        requirements = parse_requirements(requirements_file)
    if dependency_links is None:
        dependency_links = parse_dependency_links(requirements_file)
else:
    if requirements is None:
        requirements = []
    if dependency_links is None:
        dependency_links = []

# Install main directory using setuptools library, while skipping test.
packages = [package for package in packages if package != "test"]
package_data = {k: v for k, v in package_data.items() if k != "test"}
setuptools.setup(
    name=package_name,
    version=version,
    packages=packages,
    scripts=scripts,
    package_data=package_data,
    include_package_data=True,
    install_requires=requirements,
)
