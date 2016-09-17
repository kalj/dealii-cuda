#!/bin/bash
#
# @(#)bmscript.sh
# @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>


if [ "$1" == 'ball' ] ; then
    GRID='ball'
elif [ "$1" == 'cube' ] ; then
    GRID='cube'
else
    echo "invalid grid type: $1"
    exit 1
fi

if [ "$2" == "color" ] ; then
    METHOD='color'
elif [ "$2" == 'nothing' ] ; then
    METHOD='nothing'
elif [ "$2" == 'atomic' ] ; then
    METHOD='atomic'
else
    echo "invalid merge method: $2"
    exit 1
fi

if [ "$3" == "j0" ] ; then
    USE_J0=1
fi

if [ "$4" == "hn" ] ; then
    HANGING=1
fi

function comp()
{
    d=$1
    p=$2

    BDIR=build_${d}_${p}

    if [ -d $BDIR ] ;
    then
        # >&2 echo "ERROR: Directory $BDIR already exists!"
        # exit 1
        >&2 echo "Note: Directory $BDIR already exists"

    else
        mkdir $BDIR
    fi

    (
        cd $BDIR

        if [ "$GRID" == 'ball' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DBALL_GRID"
        fi

        if [ "$METHOD" == "color" ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -DMATRIX_FREE_COLOR"
        elif [ "$METHOD" == 'nothing' ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -DMATRIX_FREE_NOTHING"
        else
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p}"
        fi

        if [ "$USE_J0" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_UNIFORM_MESH"
        fi

        if [ "$HANGING" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_HANGING_NODES"
        fi


        cmake -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native --std=c++11"  -DCMAKE_BUILD_TYPE=Release ../ 2>>compile.log >>compile.log

        make bmop 2>>compile.log >>compile.log

        make bmop_spm 2>>compile.log >>compile.log
    ) &

}

function run()
{
    d=$1
    p=$2

    # max ndofs
    M=10000000

    #factor amplifying 2D problems somewhat (now 10 more DoFs are allowed in 2D)
    fact=10
    maxref=$(octave -q --eval "disp( floor( log2( (nthroot(${M}*(${fact})**(3-${d})),${d})-1)/${p} ) ) )")
    minref=$(octave -q --eval "disp(max(0,${maxref}-4 ))")


    BDIR=build_${d}_${p}
    cd $BDIR

    if [ -x ./bmop ] ; then
        echo '-- mf_gpu --'
        ./bmop ${maxref} ${minref} | tee mf_gpu_output.log
    fi

    if [ -x ./bmop_spm ] ; then
        echo '-- spm_gpu --'
        ./bmop_spm ${maxref} ${minref} | tee spm_gpu_output.log
    fi

    cd ..
}

function myloop()
{
    myfun=$1
    for d in 2 3 ; do

        for p in 1 2 3 4 ; do

            $myfun $d $p

        done

    done
}

myloop comp

wait
>&2 echo 'everything built!'

myloop run
