# This is the officially unofficial PyTorch implementation of BigGAN adapted from <https://github.com/ajbrock/BigGAN-PyTorch.git> adapted to run on the MIT Satori PowerAI Cluster

## To deploy on Satori do the following:

1. Get access to Satori following instructions in the [Satori Documentation](https://mit-satori.github.io/satori-basics.html)
2. Point your browse to the [Satori Open On-Demand (OOD)  portal](https://satori-portal.mit.edu/pun/sys/dashboard)
3. Set up and activate the [IBM Watson Machine Learning Community Edition (WMLCE)](https://mit-satori.github.io/satori-ai-frameworks.html#) conda environment.
4. On the top menu bar got to **Clusters -> Satori Shell Access**.
5. In the  shell get the test repo by typing  **git clone <https://github.com/alexandonian/BigGAN-PyTorch.git>**. Please read the README of that repo for an in-depth explanation of the steps we will complete.
6. Once the repo has been cloned, check out the `satori` branch with: \
`git checkout -b satori --track origin/satori`
7. Next, run the setup script with: \
`sh setup.sh` \
to prepare some data directories and symlinks. Currently, ImageNet is the only shared dataset stored on Satori under `/data/ImageNet`; however, more may be added in the future.
8. (Optional): To prepare your dataset as a single HDF5 file, please run \
`bsub < jobs/make_hdf5.lsf` \
with the appropriate parameters.
9. In order to measure sample quality during training, you will need to precompute inception moments for the datset of interest. To do this, run the corresponding lsf script with: \
`bsub < jobs/calculate_inception_moments.lsf` \
10. Now we are ready to submit our first training job, which can be done with any of the `jobs/biggan*` lsf scripts. For example, run \
`bsub < jobs/biggan_deep128_imagenet.lsf` \
to start training a 128px resolution BigGAN-Deep model on ImageNet.
11. During training, it's useful to monititor various training metrics, which can be done via a Jupyter Notebook. Go back to the OOD Dashboad window (labeld **My Interactive Sessions**) and go to menu option **Interactive Apps -> Jupyter Notebook**.
12. Click the **Connect to Jupyter** button when it appears in a few moments
13. When Jupyter comes up for the first time, you may be prompted to select a kernel, If so, choose the default **Python 3 PowerAI**
14. Use the left navigation pane to find the git repo directory (**BigGAN-PyTorch**) you downloaded in step 4. Click into `BigGAN-PyTorch/notebooks` and double click on the Jupyter notebook **Monitor.ipynb**.
15. **Enjoy !!!** -- but also think about the energy you are using and how you might reduce it.
