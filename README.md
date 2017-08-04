

For the application of the neural network we will be using the [Keras framework](https://keras.io). We will be running it with the [TensorFlow](https://www.tensorflow.org) backend.

The first step is installing these frameworks (and a few other ones) so they can be used.

We will be working primarily in a virtual environment and we will naturally be using root files for our samples, so it is necessary to install root with python support.
Do not forget to run the `thisroot.sh` or `thisroot.csh` script (whichever is applicable to your shell) at the start of each session or else the root executables and libraries may not be found.

You can then initialize the virtualenv with the following command:
```sh
virtualenv --system-site-packages -p [pythonVer] keras-tf
```

It may not be necessary to specify the python version, but I show here how to do it in case it is needed.
Remember, for pyROOT python2.7 is really recommended.

The last parameter in the `virtualenv` command is the name of the virtualenv and simultaneously the directory where the virtualenv will be stored.
Feel free to change it to something more adequate for your uses, but remember you choice.

Once the virtualenv is created, we can now activate it to start using it.
Once activated, any packages installed for python will be installed in the virtualenv and thus be contained to it.
To activate the virtualenv run the command:
```sh
source keras-tf/bin/activate
```

Once you are done, you can deactivate the virtualenv with the command:
```sh
deactivate
```

Remember that you must always activate the virtualenv when you want to use Tensorflow/Keras, just like the script above for root.

We can finally install all the needed packages.
For this we will use pip with the following commands:
```sh
pip install --upgrade tensorflow keras pandas scikit-learn root_numpy h5py matplotlib
```

A note about matplotlib, on a Mac it turns out that when using a virtualenv a "non-framework" version of python is created, but matplotlib kind of requires a framework version of python.
There doesn't seem to be a very good workaround at the moment of this writing, but the solution I found was to run the non-virtualenv python (i.e. the system python) while within the virtualenv.
In this way we loose a bit of the encapsulation (although not all of it), but we can easily run our scripts.
The best would be to limit the use of matplotlib to as few scripts as possible and only run those scripts with this "hack".
To simplify this non-standard usage, I have added the following lines to my `.profile`:
```sh
function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
        PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python "$@"
    else
        /usr/local/bin/python "$@"
    fi
}
```

At which point I can run the system python within the virtualenv by issuing `frameworkpython` instead of `python`


