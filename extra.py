import subprocess
import os
import json
import re
import sys
from tqdm import tqdm


def check_dependencies():
    for cmd in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Error: {cmd} not found. Please make sure it's installed and in your PATH.")
            return False
    return True


def get_video_info(file_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", file_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error determining video information: {e}")
        return None


def run_ffmpeg_with_progress(command, duration):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1,
        universal_newlines=True
    )

    pbar = tqdm(total=100)

    for line in iter(process.stdout.readline, ''):
        match = re.search(r"out_time=(\d{2}):(\d{2}):(\d{2})\.", line)
        if match:
            h, m, s = map(int, match.groups())
            current = h*3600 + m*60 + s
            progress = min(int(100 * current / duration), 100)
            pbar.n = progress
            pbar.refresh()

    pbar.close()
    process.wait()
    return process.returncode


def compress_video(input_file, output_file, target_size_mb=None):
    if not check_dependencies():
        return False
        
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return False
        
    info = get_video_info(input_file)
    if not info:
        return False
        
    duration = float(info['format'].get('duration', 0))
    if duration == 0:
        print("Could not determine video duration.")
        return False
        
    # Get original streams info
    audio_streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
    video_streams = [s for s in info['streams'] if s['codec_type'] == 'video']
    
    orig_audio_bitrate = 128000
    if audio_streams and 'bit_rate' in audio_streams[0]:
        orig_audio_bitrate = int(audio_streams[0]['bit_rate'])
        
    video_stream = video_streams[0] if video_streams else {}
    height = int(video_stream.get('height', 0))
    
    target_audio_bitrate = min(128000, orig_audio_bitrate)
    
    orig_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    
    print(f"Original file size: {orig_size_mb:.2f} MB")
    if target_size_mb and orig_size_mb <= target_size_mb:
        print(f"Original file is already smaller than or equal to {target_size_mb} MB.")
        print("Avoiding unnecessary re-encoding.")
        return True

    if target_size_mb is not None:
        target_size_bits = target_size_mb * 8 * 1024 * 1024
        target_total_bitrate = target_size_bits / duration
        video_bitrate = target_total_bitrate - target_audio_bitrate
        
        if video_bitrate <= 0:
            print(f"Error: Target size {target_size_mb}MB is too small for a video of {duration:.1f}s.")
            return False
            
        print(f"Targeting size: {target_size_mb}MB.")
        print(f"Calculated video bitrate: {video_bitrate/1000:.1f} kbps, Audio bitrate: {target_audio_bitrate/1000:.1f} kbps")
        
        # Intelligent scaling if bitrate is too low for the resolution
        scale_filter = None
        if height > 0:
            if video_bitrate < 200000 and height > 360:
                scale_filter = "scale=-2:360"
                print("Bitrate is very low; scaling down to 360p to preserve visual quality.")
            elif video_bitrate < 500000 and height > 480:
                scale_filter = "scale=-2:480"
                print("Bitrate is low; scaling down to 480p to preserve visual quality.")
            elif video_bitrate < 1000000 and height > 720:
                scale_filter = "scale=-2:720"
                print("Bitrate is moderate; scaling down to 720p to preserve visual quality.")

        passlogfile = f"ffmpeg2pass_{os.getpid()}"
        
        try:
            # Pass 1
            print("\nStarting Pass 1/2...")
            pass1_cmd = [
                "ffmpeg", "-y", "-i", input_file, 
                "-c:v", "libx264", "-b:v", str(int(video_bitrate)),
                "-pass", "1", "-passlogfile", passlogfile,
                "-an", "-f", "mp4", "-progress", "pipe:1"
            ]
            if scale_filter:
                pass1_cmd.extend(["-vf", scale_filter])
            pass1_cmd.append(os.devnull)
            
            p1_code = run_ffmpeg_with_progress(pass1_cmd, duration)
            if p1_code != 0:
                print("Error during Pass 1 encoding.")
                return False

            # Pass 2
            print("\nStarting Pass 2/2...")
            pass2_cmd = [
                "ffmpeg", "-y", "-i", input_file, 
                "-c:v", "libx264", "-b:v", str(int(video_bitrate)),
                "-pass", "2", "-passlogfile", passlogfile,
                "-c:a", "aac", "-b:a", str(target_audio_bitrate),
                "-progress", "pipe:1", "-f", "mp4"
            ]
            if scale_filter:
                pass2_cmd.extend(["-vf", scale_filter])
            pass2_cmd.append(output_file)
            
            p2_code = run_ffmpeg_with_progress(pass2_cmd, duration)
            if p2_code == 0:
                print("\nCompression completed successfully!")
                print(f"Output file size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
                return True
            else:
                print(f"\nAn error occurred during compression. Return code: {p2_code}")
                return False
                
        finally:
            # Clean up two-pass log files
            for file in os.listdir("."):
                if file.startswith(passlogfile) and file.endswith(".log") or file.endswith(".log.mbtree"):
                    os.remove(file)
                    
    else:
        # Automatic optimal settings using CRF
        print("No target size provided. Using automatic CRF compression (preserving resolution/audio).")
        cmd = [
            "ffmpeg", "-y", "-i", input_file, 
            "-c:v", "libx264", "-preset", "slower",
            "-crf", "23", 
            "-c:a", "aac", "-b:a", str(target_audio_bitrate),
            "-progress", "pipe:1",
            "-f", "mp4", output_file
        ]
        print("\nStarting compression...")
        code = run_ffmpeg_with_progress(cmd, duration)
        
        if code == 0:
            print("\nCompression completed successfully!")
            print(f"Output file size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"\nAn error occurred during compression. Return code: {code}")
            return False


if __name__ == "__main__":
    # Example Usage
    # sample_input = "file_example_MP4_1280_10MG.mp4"
    sample_input = "/home/acquaint/production/video_compressor/file_example_MP4_1280_10MG.mp4"
    if not os.path.exists(sample_input):
        print(f"Please provide the {sample_input} in the current directory.")
        sys.exit(1)
        
    print("="*40)
    print("CASE 1: Target Size Provided (5 MB)")
    print("="*40)
    compress_video(sample_input, "output_target_5MB.mp4", target_size_mb=500.0)
    
    print("\n" + "="*40)
    # print("CASE 2: Target Size Not Provided (Auto / CRF)")
    # print("="*40)
    # compress_video(sample_input, "output_auto.mp4", target_size_mb=None)
