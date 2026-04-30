#!/usr/bin/env python3
import sys
import subprocess
import shutil

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 audiotovideo.py <image> <audio>")
        sys.exit(1)

    image = sys.argv[1]
    audio = sys.argv[2]
    output = "output.mp4"

    # Ensure ffmpeg exists
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH")
        sys.exit(1)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output
        "-loop", "1",         # loop the image
        "-i", image,
        "-i", audio,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-vf", "scale=1920:1080,fps=60",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output
    ]

    print("Running ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Created {output}")
    except subprocess.CalledProcessError:
        print("ffmpeg failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

