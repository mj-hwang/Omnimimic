from setuptools import setup, find_packages

setup(
    name="omnimimic",
    packages=[
        package for package in find_packages() if package.startswith("omnimimic")
    ],
    description="Omnimimic : Wrappers and pipeline scripts for collecting data and training imitation learning (IL) agents in the Omnigibson environment under the robomimic framework.",
)