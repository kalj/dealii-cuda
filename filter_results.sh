#!/bin/bash

for m in cube ball  ; do
#    for h in nohn hn ; do
        for c in color atomic ; do
#            dir=results_${m}_${c}_${h}
            dir=results_${m}_${c}_hn_adaptive
            # r=${dir}.txt

            for b in ${dir}/build_* ; do
                for t in mf spm ; do
                    if [ $t == "mf" ] ; then
                        meth="mf_${c}"
                    elif [ $c == "color" ] ; then
                        meth="spm"
                    else
                        # skip duplicate entry
                        continue
                    fi

                    grep '^[23]' ${b}/${t}_gpu_output.log | awk "{printf \"%10s%10d\t%10d\t%10d\t%10g\n\", \"${meth}\", \$1, \$2, \$3, \$4; }"
                done
            done

        done | sort -k1,1 -k2,2 -k3,3 -k4,4g  > results_${m}_adaptive.txt
    # done
done
