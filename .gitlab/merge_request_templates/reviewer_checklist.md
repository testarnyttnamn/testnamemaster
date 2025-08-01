## MR Description

Closes #`<ISSUE NUMBER>`

## MR Checklist for Reviewers

Reviewers, please check the following points before merging.

- [ ] MR targets the correct branch
- [ ] CI tests have run and passed for the latest commit on the source branch
- [ ] Check that the code can still be installed if new packages are imported
- [ ] If necessary, the notebooks in `cloe/tests/test_tools/notebooks` have been run to generate the pickle files required for the unit tests
- [ ] Coverage percentage is retained or increased
- [ ] Quality of new/changed code is acceptable
- [ ] Quality of new/changed unit tests is acceptable
- [ ] No data files have been included in the commits
- [ ] All Jupyter notebook cells have been cleared 
- [ ] Manual `Check_HTML` job run and API documentation looks good
- [ ] Manual `Profiling` job run and no unexpected performances issues found
- [ ] Manual `Verification` job run and results match expectations
- [ ] The `internal/Scripts_and_options.ipynb` notebook has been run if the MR includes changes to configuration files (i.e. the `.yaml` files in `configs`)
- [ ] Implementation follows the agreed task description point by point
- [ ] Check that there are no '\No newline at the end of file' warnings
- [ ] Check that any added folder/file has been added to the `README.md` file
- [ ] Check that the implementation follows the [Contributing guidelines and style choices](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/develop/CONTRIBUTING.md)
- [ ] At least one expert has verified the documentation
- [ ] Final comments by leads, if any, are correctly taken into account (minor ones within the task; major ones in a new task)
- [ ] Check that the corresponding branch has been deleted after merging. If not, delete it

> Refer to [reviewer guidelines](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/wikis/Guidelines-for-Reviewers) for more information.
