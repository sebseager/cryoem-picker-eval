This directory contains the bulk of the statistical analysis pipeline for this study, as well as a number of auxiliary/helper scripts.

## Pipeline

To start processing an annotated cryo-EM dataset (one with both micrograph images and accompanying ground-truth coordinate files) with this pipeline, follow the steps below. To download a published cryo-EM dataset from the EMPIAR database, please take a look at our [guide to EMPIAR](docs/empiar.md).

1. Run `coord_converter.py` with `--require_conf` and `-t box` to generate `.box` files from whichever format the ground-truth coordinate files were provided as. The `-f` flag will need to be set differently based on the input file format.

2. Run `write_filematching.py` to match micrographs and ground-truth coordinate files for all pickers. The script requires a `--primary_key` argument, which indicates the name of the file set we want to use to build the match. Other arbitrary keyword arguments also need to be passed in, corresponding to your desired file sets (micrographs, ground-truth coordinates, coordinates for a particle picker, etc.) An example usage follows:

   ```bash
   python write_filematching.py out_dir/ --primary_key mrc --mrc path/to/*.mrc --gt path/to/*.box --picker1 path/to/*.box --picker2 path/to/*.box
   ```

3. Run `write_jaccard.py` to make many-to-one and max-bipartite matching tables between ground-truth boxes and all picker boxes. For the most part it will probably be unnecessary to modify defaults for optional keyword arguments.
