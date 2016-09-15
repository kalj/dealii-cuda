#!/bin/bash
#
# @(#)bmscript.sh
# @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>

if [ "$1" == 'pmem' ] ; then
    PARALLELIZATION='pmem'
elif [ "$1" == 'shmem' ] ; then
    PARALLELIZATION='shmem'
else
    echo "invalid parallelization method: $1"
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

function comp()
{
    d=$1
    p=$2
    descr=$3
    bksize=$4

    if [ "$PARALLELIZATION" == 'pmem' ] ; then
        BDIR=build_${d}_${p}_${bksize}
    else
        BDIR=build_${d}_${p}_${descr}_${bksize}
    fi

    if [ -d $BDIR ] ;
    then
        >&2 echo "ERROR: Directory $BDIR already exists!"
        exit 1
    fi
    mkdir $BDIR
    (
        cd $BDIR

        if [ "$METHOD" == "color" ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -march=native -DMATRIX_FREE_BKSIZE_APPLY=${bksize} -DMATRIX_FREE_COLOR --std=c++11"
        elif [ "$METHOD" == 'nothing' ] ; then
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -march=native -DMATRIX_FREE_BKSIZE_APPLY=${bksize} -DMATRIX_FREE_NOTHING --std=c++11"
        else
            CMAKE_CXX_FLAGS=" -DDIMENSION=${d} -DDEGREE_FE=${p} -march=native -DMATRIX_FREE_BKSIZE_APPLY=${bksize} --std=c++11"
        fi

        if [ "$PARALLELIZATION" == 'shmem' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_PAR_IN_ELEM"
        fi

        if [ "$USE_J0" == '1' ] ; then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMATRIX_FREE_J0"
        fi


        cmake -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"  -DCMAKE_BUILD_TYPE=RELEASE -D DEAL_II_DIR=/local/home/karll/sw/deal.II ../ 2>>compile.log >>compile.log

        make bmop 2>>compile.log >>compile.log
    ) &

}

function run()
{
    d=$1
    p=$2
    descr=$3
    bksize=$4

    if [ "$PARALLELIZATION" == 'pmem' ] ; then
        BDIR=build_${d}_${p}_${bksize}
        printf "%5d %5d %5d %9s\t" ${d} ${p} ${bksize}
    else
        BDIR=build_${d}_${p}_${descr}_${bksize}
        printf "%5d %5d %5d %9s\t" ${d} ${p} ${bksize} $descr
    fi
    cd $BDIR


    if [ -x ./bmop ] ; then
        ./bmop > output.log
        cat output.log | sed -n 's!.*Per iteration \(.*\)s$!\1!p'

    else
        echo '-'
    fi
    cd ..
}

function myloop()
{
    myfun=$1
    for d in 2 3 ; do

        for p in 1 2 3 4 ; do

            if [ "$PARALLELIZATION" == 'shmem' ] ; then
                T=$(($p+1))
                NTS=$(( $T**$d ))

                if [ $d == 3 ] ; then
                    descr="[$T,$T,$T]"
                else
                    descr="[$T,$T]"
                fi

                bksize=32
                for ((bksize=1 ; bksize*$NTS <= 320 ; bksize *= 2)) ; do
                    $myfun $d $p $descr $bksize
                done
            else

                for ((bksize=16 ; bksize <= 64 ; bksize *= 2)) ; do
                    $myfun $d $p apa $bksize
                done
            fi

        done
        wait

    done
}

myloop comp

wait
>&2 echo 'everything built!'

myloop run
