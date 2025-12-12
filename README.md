# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pylhc/ir_amplitude_detuning/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                       |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------- | -------: | -------: | ------: | --------: |
| examples/\_\_init\_\_.py                                   |        0 |        0 |    100% |           |
| examples/commissioning\_2022.py                            |       60 |        0 |    100% |           |
| examples/md3311.py                                         |       65 |        0 |    100% |           |
| examples/md6863.py                                         |      170 |        3 |     98% |125, 210, 232 |
| ir\_amplitude\_detuning/\_\_init\_\_.py                    |        7 |        0 |    100% |           |
| ir\_amplitude\_detuning/detuning/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| ir\_amplitude\_detuning/detuning/calculations.py           |       61 |        0 |    100% |           |
| ir\_amplitude\_detuning/detuning/equation\_system.py       |      120 |        0 |    100% |           |
| ir\_amplitude\_detuning/detuning/measurements.py           |      235 |        4 |     98% |104, 131, 373-374 |
| ir\_amplitude\_detuning/detuning/targets.py                |       27 |        0 |    100% |           |
| ir\_amplitude\_detuning/detuning/terms.py                  |       22 |        0 |    100% |           |
| ir\_amplitude\_detuning/lhc\_detuning\_corrections.py      |      147 |        2 |     99% |   423-424 |
| ir\_amplitude\_detuning/plotting/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| ir\_amplitude\_detuning/plotting/correctors.py             |       95 |        2 |     98% |   88, 111 |
| ir\_amplitude\_detuning/plotting/detuning.py               |      120 |        1 |     99% |       118 |
| ir\_amplitude\_detuning/plotting/utils.py                  |       43 |        0 |    100% |           |
| ir\_amplitude\_detuning/simulation/\_\_init\_\_.py         |        0 |        0 |    100% |           |
| ir\_amplitude\_detuning/simulation/lhc\_simulation.py      |      153 |        9 |     94% |157, 162, 257-260, 264, 283, 385 |
| ir\_amplitude\_detuning/simulation/results\_loader.py      |       62 |        1 |     98% |       171 |
| ir\_amplitude\_detuning/utilities/\_\_init\_\_.py          |        0 |        0 |    100% |           |
| ir\_amplitude\_detuning/utilities/common.py                |       62 |        0 |    100% |           |
| ir\_amplitude\_detuning/utilities/constants.py             |       12 |        0 |    100% |           |
| ir\_amplitude\_detuning/utilities/correctors.py            |       68 |        1 |     99% |        50 |
| ir\_amplitude\_detuning/utilities/latex.py                 |       51 |        1 |     98% |        36 |
| ir\_amplitude\_detuning/utilities/logging.py               |        5 |        0 |    100% |           |
| ir\_amplitude\_detuning/utilities/measurement\_analysis.py |       75 |        0 |    100% |           |
|                                                  **TOTAL** | **1660** |   **24** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pylhc/ir_amplitude_detuning/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pylhc/ir_amplitude_detuning/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pylhc/ir_amplitude_detuning/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/ir_amplitude_detuning/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpylhc%2Fir_amplitude_detuning%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/ir_amplitude_detuning/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.