#!/bin/bash
#
alpha=$1
#
homedir='/lfs3/projects/sso/Chris.Rozoff/h2'
datadir='/lfs3/projects/sso/Chris.Rozoff/anen/realtime/hwrf_files'
workdir=$homedir'/tes_out'
tempdir=$workdir'/'$alpha'/'
gribfile='gfile2'
#
if [ ! -d $tempdir ]; then
   mkdir $tempdir
fi
#
cd $tempdir
#
filestr=$alpha'/*hwrfprs.core*'
#
ls $datadir'/'$filestr > $alpha'_input_list.dat'

fname0=`cat $alpha'_input_list.dat'`
sdate0=`cat $alpha'_input_list.dat' | awk -F . '{print $3}'`
sname0=`cat $alpha'_input_list.dat' | awk -F . '{print $2}' | awk -F / '{print $6}'`
fhr0=`cat $alpha'_input_list.dat' | awk -F . '{print $7}'`
nel1=`echo $sdate0 | wc -w`
#
echo $nel1 files to be processed
#
np=1
for fname in $fname0
do
  fname1[$np]=`echo $fname`
  let np=np+1
done
np=1
for sname in $sname0  
do
  sname1[$np]=`echo $sname`  
  let np=np+1
done
np=1
for sdate in $sdate0
do
  sdate1[$np]=`echo $sdate`  
  let np=np+1
done
np=1
for fhr in $fhr0
do
  fhr1[$np]=`echo $fhr` 
  let np=np+1 
done
np=1
while [ $np -le $nel1 ]
 do
 if test -r ${sname1[$np]}'.'${sdate1[$np]}'.'${fhr1[$np]}'.nc'
  then
    echo complete
  else
    echo ${sname1[$np]}'.'${sdate1[$np]}'.'${fhr1[$np]}'.nc' incomplete
    echo ${fname1[$np]}
    ln -sf ${fname1[$np]} ./$gribfile

    wgrib2 gfile2 -match ':(UGRD|VGRD|TMP|RH):(850|700|500|200) mb|:(UGRD|VGRD):(10) m above ground|:(TMP|RH):(2) m above ground|:(PRES):(surface)' -grib temp_file
    wait

    ncl_convert2nc temp_file -e grb2
    wait

    mv temp_file.nc '../'${sname1[$np]}'.'${sdate1[$np]}'.'${fhr1[$np]}'.nc' 

    rm temp_file

    echo $np >> $homedir/${alpha}_nc_vars.complete
  fi

 let np=np+1
 done


exit 0
