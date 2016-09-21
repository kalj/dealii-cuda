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

if [ "$5" == "adaptive" ] ; then
    ADAPTIVE=1
fi


if [ $ADAPTIVE == 1 ] && [ $HANGING != 1 ] ; then
    echo "it doesn't make sense with adaptivity without hanging node treatment!"
    exit 1
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


        if [ "$METHOD" == "color" ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -DMATRIX_FREE_COLOR"
        elif [ "$METHOD" == 'nothing' ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -DMATRIX_FREE_NOTHING"
        else
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p}"
        fi

        if [ "$GRID" == 'ball' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DBALL_GRID"
        fi

        if [ "$USE_J0" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_UNIFORM_MESH"
        fi

        if [ "$HANGING" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_HANGING_NODES"
        fi

        if [ "$ADAPTIVE" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DADAPTIVE_GRID"
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


    BDIR=build_${d}_${p}
    cd $BDIR

    if [ -x ./bmop ] ; then

        if [ "${d}" == 2 ] ; then

            if [ "${p}" == 1 ] ; then
                maxref=12
            elif [ "${p}" == 2 ] ; then
                maxref=11
            elif [ "${p}" == 3 ] ; then
                maxref=11
            elif [ "${p}" == 4 ] ; then
                maxref=11
            fi
        else
            if [ "${p}" == 1 ] ; then
                maxref=7
            elif [ "${p}" == 2 ] ; then
                maxref=7
            elif [ "${p}" == 3 ] ; then
                maxref=6
            elif [ "${p}" == 4 ] ; then
                maxref=6
            fi
        fi

        if [ "$GRID" == "ball" ] ; then
            let maxref=maxref-1
        fi

        minref=$(octave -q --eval "disp(max(0,${maxref}-4 ))")

        echo '-- mf_gpu --'
        ./bmop ${maxref} ${minref} | tee mf_gpu_output.log
    fi

    if [ -x ./bmop_spm ] ; then


        if [ "${d}" == 2 ] ; then

            if [ "${p}" == 1 ] ; then
                maxref=12
            elif [ "${p}" == 2 ] ; then
                maxref=11
            elif [ "${p}" == 3 ] ; then
                maxref=10
            elif [ "${p}" == 4 ] ; then
                maxref=10
                # if using hyb format:
                # maxref=9
            fi
        else
            if [ "${p}" == 1 ] ; then
                maxref=7
            elif [ "${p}" == 2 ] ; then
                maxref=6
            elif [ "${p}" == 3 ] ; then
                maxref=5
            elif [ "${p}" == 4 ] ; then
                maxref=5
            fi
        fi

        if [ "$GRID" == "ball" ] ; then
            let maxref=maxref-1
        fi


        minref=$(octave -q --eval "disp(max(0,${maxref}-4 ))")


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
