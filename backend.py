from flask import Flask
from inference.run_on_video import run_on_video
import re
from pathlib import Path
from flask import send_file, send_from_directory
from flask import Flask, request, jsonify, Blueprint
import os


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @image_mask_bp.route('/image-mask', methods=['POST'])
# def upload_image_mask_pairs():
#     if 'images' not in request.files or 'masks' not in request.files:
#         return jsonify({'error': 'Missing images or masks files'}), 400
    
#     images = request.files.getlist('images')
#     masks = request.files.getlist('masks')
#     if len(images) != len(masks):
#         return jsonify({'error': 'Number of images and masks must match'}), 400
#     if len(images) == 0 or len(masks) == 0:
#         return jsonify({'error': 'No files selected'}), 400
#     base_dir, batch_id = create_user_directories('image_mask')
#     images_dir = os.path.join(base_dir, 'images')
#     masks_dir = os.path.join(base_dir, 'masks')    
#     saved_files = []
#     for i, (image, mask) in enumerate(zip(images, masks)):
#         if not (image and allowed_file(image.filename) and mask and allowed_file(mask.filename)):
#             continue
#         image_filename = secure_filename(image.filename)
#         mask_filename = secure_filename(mask.filename)
#         if os.path.splitext(image_filename)[0] != os.path.splitext(mask_filename)[0]:
#             base_name = f"pair_{i}"
#             image_ext = os.path.splitext(image_filename)[1]
#             mask_ext = os.path.splitext(mask_filename)[1]
            
#             image_filename = f"{base_name}{image_ext}"
#             mask_filename = f"{base_name}{mask_ext}"
    
#         os.makedirs(images_dir, exist_ok=True)
#         os.makedirs(masks_dir, exist_ok=True)
        
#         image_path = os.path.join(images_dir, image_filename)
#         mask_path = os.path.join(masks_dir, mask_filename)
        
#         image.save(image_path)
#         mask.save(mask_path)
        
#         saved_files.append({
#             'image': image_filename,
#             'mask': mask_filename
#         })
    
#     return jsonify({
#         'success': True,
#         'batch_id': batch_id,
#         'cache_directory': base_dir,
#         'files': saved_files
#     }), 200


# app.register_blueprint(image_mask_bp)
# app.register_blueprint(images_bp)


# @app.route('/list-caches', methods=['GET'])
# def list_cache_directories():
#     user_cache_dir = tempfile.gettempdir()
#     directories = [d for d in os.listdir(user_cache_dir) 
#                   if os.path.isdir(os.path.join(user_cache_dir, d)) and 
#                   (d.startswith('image_mask_') or d.startswith('images_only_'))]
    
#     return jsonify({
#         'cache_directories': directories
#     }), 200


def inference():
    masks_path = "/mnt/d/xmem2/XMem2/example_videos/chair/Annotations" 
    output_path = "output_dir/"
    video_path = "/mnt/d/xmem2/XMem2/example_videos/chair/chair.mp4"
    frames_with_masks = []
    for file_path in (p for p in Path(masks_path).iterdir() if p.is_file()):
        frame_number_match = re.search(r'\d+', file_path.stem)
        if frame_number_match is None:
            print(f"ERROR: file {file_path} does not contain a frame number. Cannot load it as a mask.")
            exit(1)
        frames_with_masks.append(int(frame_number_match.group()))
    
    print("Using masks for frames: ", frames_with_masks)

    p_out = Path(output_path)
    p_out.mkdir(parents=True, exist_ok=True)
    run_on_video(video_path, masks_path, output_path, frames_with_masks)
    return output_path

@app.route('/inference')
def infer():
    file_path = inference()
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
