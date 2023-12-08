# Loop-Finder-Py

Loop-Finder-py is a command-line program that finds and saves "looping" gifs from a video. There is a web version whose demo and source code can be found below; it has a limited parameter selection and only runs on a small, selected portion of the video.

Web Demo: https://loop-finder.vercel.app (Try using Chrome if you run into issues of it not loading)\
Source Code: https://github.com/jcf1/Loop-Finder

## Usage

```
loop-finder.py [-h] filepath [-result RESULT_FOLDER] [-min MIN_LEN] [-max MAX_LEN] [-start START] [-end END] [-threshold THRESHOLD] [-eval EVAL] [-hash_func HASH_FUNC] [-clip CLIP] [-fps FPS] [-hash HASH] [-bands BANDS] 
```

| Argument | Description |
| --- | --- |
| -h, --help | Show a help/usage message and exit. |
| filepath | Video file with path included. (required) |
| -result RESULT_FOLDER | Folder to store resulting gifs, if found, for this run. Deafult is "./gif_output". |
| -min MIN_LEN | Minimum length of resulting gif in seconds. Default is 0.5, minimum is 0.01. |
| -max MAX_LEN | Maximun length of resulting gif in seconds. Default is 10. |
| -start START | When in the video to start looking for loops in seconds. Default is 0. |
| -end END | When in the video to stop looking for gifs in seconds or -1 for end of video. Default is -1. |
| -threshold THRESHOLD | Minimum threshold score a gif must attain to qualitfy as a valid loop. Default is 85. |
| -eval EVAL | Metric used to order gifs when generating results (results cannot overlap). Can be either "quality" or "length". Default is "qualify". |
| -hash_func HASH_FUNC | Hashing function to use when scoring potential results. Can be "ahash", "phash", "dhash", or "whash". Default is dhash. More info on these functions can be found here: https://github.com/JohannesBuchner/imagehash |
| -clip CLIP | Length each section of the video will be broken up into when being processed. Should only be changed if there are memory issues. Default is 60. |
| -fps FPS | Number of frames per seonds that will be reviewed for potential loops. Default in 100. |
| -hash HASH | Size that we resize the image to when using our hash function. Default is 64. |
| -bands BANDS | Number of potential bands an image can fall into based on its hash value. Default it 32. |
