import subprocess
import sys
import os

def create_video(image_path, audio_path, output_path="output.mp4"):
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    print(f"🎬 Processing: {image_path} + {audio_path} -> {output_path}")

    # FFmpeg command breakdown:
    # -loop 1: Loop the single image
    # -i image: Input image
    # -i audio: Input audio
    # -c:v libx264: Use H.264 codec
    # -tune stillimage: Optimize encoding for a non-moving image
    # -c:a aac: Encode audio to AAC
    # -b:a 192k: Set audio bitrate
    # -pix_fmt yuv420p: Ensure compatibility with most video players
    # -vf: Video filter to scale to 1080p while maintaining aspect ratio (adds black bars if needed)
    # -r 60: Set frame rate to 60fps
    # -shortest: Stop encoding when the audio stream ends
    
    command = [
        'ffmpeg',
        '-loop', '1',
        '-i', image_path,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-tune', 'stillimage',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-vf', "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        '-r', '60',
        '-shortest',
        output_path,
        '-y'  # Overwrite output file if it exists
    ]

    try:
        subprocess.run(command, check=True)
        print(f"\n✅ Success! Video saved as: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred while running FFmpeg: {e}")
    except FileNotFoundError:
        print("\n❌ Error: FFmpeg is not installed or not in your PATH.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 audiotovideo.py <image_file> <audio_file>")
    else:
        img_in = sys.argv[1]
        aud_in = sys.argv[2]
        create_video(img_in, aud_in)
