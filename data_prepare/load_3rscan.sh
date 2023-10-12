data_path=/data1/sgl/3RScan
cd $data_path
files=$(ls $folder)
for file in $files
do
    echo $file 
    unzip $data_path/$file/sequence.zip -d $data_path/$file/sequence
done