# Particle picker modifications

We made the following modifications to particle pickers to allow them to run successfully on our systems.

## ASPIRE-APPLE picker

- Classifier probabilities are not written to output .star files by default
- In helper.py, query block sizes are incorrect if either dimension of the micrograph (in pixels) is divisible by the block size.

## PARSED

- Unintended behavior is observed in autopick_columns.star output files, since Python list values were not comma-separated.
- The x and y coordinates produced in output _.star files are reversed with respect to those columns in corresponding _\_param.star files.

## DeepPicker

- We encountered a bug that occurred for datasets with low particle counts, due to the fact that DeepPicker uses a hard-coded model batch size ([see the definition of `model_input_size` here](https://github.com/nejyeah/DeepPicker-python/blob/master/train.py#L49)). To address this bug when we have a limited number of examples, we have reduced the batch size (from 100 to 32) and increased the allowed number of epochs (from 200 to 300). We get relatively the same training/validation error and test set performance (evaluated using F1 score) running DeepPicker on EMPIAR-10017 with these parameters as with the default.
- Setting `particle_number` to `-1` may result in the last example in the particle set being lost (i.e., a `train_number` of `-1` on [this line of `dataLoader.py`](https://github.com/nejyeah/DeepPicker-python/blob/3f46c8b0ffe2dbaa837fd9399b4a542588e991e6/dataLoader.py#L638) would evaluate to `array[:-1]`). To avoid this, we have set this array slice to use a very large integer value (`999999999`) as the `particle_number`.
- By default, DeepPicker randomly samples 10% of training data to be used for validation ([see `validation_ratio = 0.1` by default, here](https://github.com/nejyeah/DeepPicker-python/blob/3f46c8b0ffe2dbaa837fd9399b4a542588e991e6/dataLoader.py#L644)). We modified `train.py` to use independently provided training/validation sets to work with our picker evaluation pipeline.
