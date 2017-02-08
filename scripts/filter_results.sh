#!/bin/bash

for m in cube ball  ; do
    for h in nohn hn ; do

        for ad in adaptive noadaptive ; do

            utfil="results_${m}_${ad}_${h}.txt"

            for c in color atomic ; do
                dir=results_${m}_${c}_${h}_${ad}
                if [ -d $dir ] ; then
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

                            f="${b}/${t}_gpu_output.log"

                            if [ -f $f ] ; then
                                grep '^[23]' $f | awk "{printf \"%10s%10d\t%10d\t%10d\t%10g\n\", \"${meth}\", \$1, \$2, \$3, \$4; }"
                            fi
                        done
                    done
                fi

            done | sort -k1,1 -k2,2 -k3,3 -k4,4g  > $utfil

            if [ ! -s $utfil ]; then
                rm $utfil
            fi
        done

    done
done
