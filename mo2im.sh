set mpath $argv[1]
set fps $argv[2]
set mm (string split -rm2 / $mpath)
set mname (string split . $mm[3])


set path $mm[1]/images/$mname[1]
echo $fps
if test -z $fps
       ffmpeg -i $mpath
else if test -d (readlink -f $path)
	echo exists $path
else
	sudo mkdir $path
	echo make directory $path
	sudo ffmpeg -i $mpath -r $fps $path/image-%05d.png
end
