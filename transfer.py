from neural_style_transfer import neural_style_transfer




def transfe(content_images,style_images,progress):
    optimization_config = dict()
    optimization_config['content_weight']=1e5    
    optimization_config['style_weight']=3e4 
    optimization_config['tv_weight']=1e0
    optimization_config['height']=1000
    optimization_config['content_images_dir'] = content_images#'./dancing.jpg'
    optimization_config['style_images_dir'] = style_images#'./vane.jpg'
    # optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] =(4, '.jpg')
    optimization_config['bar']=progress
    results_path = neural_style_transfer(optimization_config)
    
    return results_path 
