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

if [ "$2" == "adaptive" ] ; then
    ADAPTIVE=1
fi



function comp()
{
    d=$1
    p=$2

    BDIR=build_${d}_${p}

    if [ -d $BDIR ] ;
    then
        >&2 echo "Note: Directory $BDIR already exists"

    else
        mkdir $BDIR
    fi

    (
        cd $BDIR


        CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p}"

        if [ "$GRID" == 'ball' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DBALL_GRID"
        fi

        if [ "$ADAPTIVE" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DADAPTIVE_GRID"
        fi


        cmake -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native --std=c++11"  -DCMAKE_BUILD_TYPE=Release ../ 2>>compile.log >>compile.log

        make bmop-cpu 2>>compile.log >>compile.log

    ) &

}

function run()
{
    d=$1
    p=$2


    BDIR=build_${d}_${p}
    cd $BDIR

    if [ -x ./bmop-cpu ] ; then

        if [ "${d}" == 2 ] ; then

            if [ "${p}" == 1 ] ; then
                maxref=12
            elif [ "${p}" == 2 ] ; then
                maxref=11
            elif [ "${p}" == 3 ] ; then
                maxref=11
            elif [ "${p}" == 4 ] ; then
                maxref=10
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

        unbuffer ./bmop-cpu ${maxref} ${minref} | tee cpu_output.log
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
