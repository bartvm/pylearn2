#/bin/sh

set -x -e

if [ "x$TEST_DOC" = "xYES" ]; then
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    python ./doc/scripts/docgen.py --test
else
    # We can't build the test dataset as the original is not
    # present. We can't download the original as it is too big to
    # download each time. If present run: python make_dataset.py
    (cd pylearn2/scripts/tutorials/grbm_smd && wget http://www.iro.umontreal.ca/~lisa/datasets/cifar10_preprocessed_train.pkl)
    if [ $TRAVIS_PYTHON_VERSION = '3.4' ]; then python pylearn2/devtools/convert_pkl.py pylearn2/scripts/tutorials/grbm_smd/cifar10_preprocessed_train.pkl; fi
    THEANO_FLAGS="$FLAGS",blas.ldflags="-lblas -lgfortran",-warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise theano-nose -v --with-coverage --cover-package=pylearn2 $PART
fi
