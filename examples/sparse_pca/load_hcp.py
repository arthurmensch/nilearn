from os.path import expanduser

from nilearn_sandbox.datasets.fetch_hcp import fetch_hcp_rest

from nilearn.input_data import MultiNiftiMasker

dataset = fetch_hcp_rest(n_subjects=1, data_dir=expanduser('~/data'))
masker = MultiNiftiMasker(mask_img=dataset.mask).fit()
masker.transform(dataset.func[0])
