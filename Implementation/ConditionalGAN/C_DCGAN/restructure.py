
class_names = [ 'abstract', 'cityscape', 'landscape', 'portrait' ]
root_dir = './Dataset/wikiart'

if os.path.isdir(os.path.join(root_dir, class_names[0])):
    os.makedirs(os.path.join(root_dir, 'test'), exist_ok = True)
    
    for i, d in enumerate(class_names):
        print(i, d)
        
    for c in class_names:
        os.makedirs(os.path.join(root_dir, 'test', c), exist_ok=True)
        
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.endswith(('.jpg', '.jpeg'))]
        sample_images = random.sample(images, 30)
        
        for image in sample_images:
            src_path = os.path.join(root_dir, c, image)
            trg_path = os.path.join(root_dir, 'test', c , image)
            shutil.move(src_path, trg_path)
            
            print(f'{len(image)}')