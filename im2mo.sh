set filename $argv[1]
set fps $argv[2]

ffmpeg -r $fps -i output/images/$filename/image-%05d.png -vcodec libx264 -r $fps output/movies/$filename.mp4

ffmpeg -i input/movies/$filename.wmv -vn -acodec copy output/sound.wav
ffmpeg -i output/movies/$filename.mp4 -i output/sound.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output/movies/aaa.mp4
sudo rm output/sound.wav output/movies/$filename.mp4
sudo mv output/movies/aaa.mp4 output/movies/$filename.mp4
