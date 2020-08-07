# extma
Extraction of LA-ICP-MS data from microarrays or spotted tissue.
Data is masked using the selected threshold method, cleaned up and large objects split using watershed segementation.

#Installation

Run `pip install -e .` in the root directory to install.

The script can then be run using `extma <path to npz> <size of core> [<options>]`.

For a list of options access the help `extma -h`.
