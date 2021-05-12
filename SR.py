from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
from SinGAN.models import SR
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from skimage import io as img

def get_pixel_data_from_image(img, base_img):
    img = img.squeeze(0).permute(1, 2, 0)
    base_img = base_img.squeeze(0).permute(1, 2, 0)
    coords = []
    targets = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = list(img[i, j].cpu().numpy())
            scaled_i = i * (base_img.shape[0] - 1) / (img.shape[0] - 1)
            scaled_j = j * (base_img.shape[1] - 1) / (img.shape[1] - 1)

            coords.append([scaled_i, scaled_j])
            targets.append(color)
    
    return torch.from_numpy(np.array(coords)), torch.from_numpy(np.array(targets))


def get_SR_inputs_targets(images):
    coords, targets = tuple(zip(*[get_pixel_data_from_image(img, images[0]) for img in images]))
    
    return torch.cat(coords).unsqueeze(1), torch.cat(targets).unsqueeze(1)
    

def predict(model, input_tensor):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128

    dataset = TensorDataset(input_tensor.float().to(device))
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)

    model.to(device)
    model.eval()

    inputs = []
    losses = []
    running_loss = 0.0
    outputs = []

    for i, data in tqdm.tqdm(enumerate(dataloader, 0)):
        
        input = data[0].to(device)
        inputs.append(input)

        with torch.no_grad():
            output = model(input)
        outputs.append(output)
        
        losses.append(running_loss/(i+1))

    # predictions = torch.cat(outputs).squeeze(1).cpu().numpy()
    return torch.cat(outputs), torch.cat(inputs)



def create_image_from_output(output_tensor, h, w):
    x = torch.zeros(1, 3, h, w)

    for i in range(0, h):
        for j in range(0, w):
            t = w * i + j
            x[0, :, i, j] = output_tensor[t, 0, :]
    
    return x
        


def predict_image(model, target_h, target_w, base_img, output_file_name = 'sr_img_output.png'):
    # base_img = base_img.squeeze(0).permute(1, 2, 0)
    coords = []
    scaled_coords = []

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for i in range(target_h):
        for j in range(target_w):
            coords.append([i, j])
            scaled_coords.append([i * (base_img.shape[2] - 1) / (target_h - 1), j * (base_img.shape[3] - 1) / (target_w - 1)])

    input_tensor = torch.from_numpy(np.array(scaled_coords)).unsqueeze(1).float().to(device)

    pred, inp = predict(model, input_tensor)
    reshape_size = list(pred.permute(1, 2, 0).shape[0:2]) + [target_h, target_w]
    pred_img = pred.permute(1, 2, 0).view(*reshape_size)
    # pred_img = create_image_from_output(pred, target_h, target_w)
    
    plt.figure(figsize = (8, 15))

    img_plot = functions.convert_image_np(pred_img)
    # plt.imshow(img_plot)
    plt.imsave(output_file_name, img_plot, vmin=0, vmax=1)
    plt.show()


def get_edge_neighbors(edge_px, sr_factor, target_h, target_w):
    thick_edge_px = []

    for x, y in edge_px:
        for i in range(-sr_factor, sr_factor):
            for j in range(-sr_factor, sr_factor):
                if (0 <= x + i < target_h) and (0 <= y + j < target_w):
                    thick_edge_px.append((x + i, y + j))
    
    return thick_edge_px

def edgeSR_generate(img_path, sr_factor, model, target_h, target_w, base_img, output_file_name = 'edge_SR.png'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img1 = img.imread(img_path)

    edge_img, edge_px = edge_detector(img_path, sr_factor, t1=50, t2=100, blur_first=False, blur_kernel_size=(2,2))
    # hr_pixels = get_HR_edge_pixels(edge_px, sr_factor)
    edge_px = get_edge_neighbors(edge_px, sr_factor, target_h, target_w)
    hr_pixels = np.array(edge_px)
    hr_pixels[:, 0] = (base_img.shape[2] - 1) * hr_pixels[:, 0] / (target_h - 1)
    hr_pixels[:, 1] = (base_img.shape[3] - 1) * hr_pixels[:, 1] / (target_w - 1)

    input_tensor = torch.from_numpy(hr_pixels).unsqueeze(1).float().to(device)

    pred, inp = predict(model, input_tensor)
    
    # inp = inp.squeeze(1).cpu().numpy()
    # inp = hr_pixels
    pred = pred.squeeze(1).cpu().numpy()

    # inp[:, 0] = inp[:, 0] * (target_h - 1) / base_img.shape[2]
    # inp[:, 1] = inp[:, 1] * (target_w - 1) / base_img.shape[3]
    # inp = inp.astype(int)
    # inp[:, 0] = np.clip(inp[:, 0], 0, target_h - 1)
    # inp[:, 1] = np.clip(inp[:, 1], 0, target_w - 1)

    img1 = cv2.resize(img1, dsize=(target_w, target_h))
    img1 = img1 / 255
    img1 = (img1 - 0.5) * 2
    img1 = np.clip(img1, -1, 1)

    hr_pixels = np.array(edge_px)
    for i in range(pred.shape[0]):
        hr_x, hr_y = tuple(hr_pixels[i, :])
        color = pred[i, :]

        img1[hr_x, hr_y] = color
    
    img1 = (img1 + 1) / 2
    img1 = np.clip(img1, 0, 1)
    
    # img1 = torch.from_numpy(img1).to(device).permute(2, 0, 1).unsqueeze(0)
    # img_plot = functions.convert_image_np(img1)
    plt.imsave(output_file_name, img1, vmin=0, vmax=1)
    # plt.show()


def edgeSR_merge_images(orig_img_path, sr_img_path, sr_factor):
    edge_img, edge_px = edge_detector(orig_img_path, sr_factor = sr_factor, t1=50, t2=100, blur_first=False, blur_kernel_size=(2,2))
    edge_px = get_edge_neighbors(edge_px, sr_factor, edge_img.shape[0], edge_img.shape[1])
    orig_img = cv2.imread(orig_img_path)
    sr_img = cv2.imread(sr_img_path)
    scaled_img = cv2.resize(orig_img, dsize=(orig_img.shape[1] * sr_factor, orig_img.shape[0] * sr_factor))

    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

    for x, y in edge_px:
        scaled_img[x, y] = sr_img[x, y]
    
    plt.imsave(f'edgeSR_merged_{sr_factor}x.png', scaled_img, vmin=0, vmax=1)
    # plt.show()



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', default="33039_LR.png")#required=True)
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--mode', help='task to be done', default='SR')
    parser.add_argument('--sr_epochs', help='SR epochs', default=1000)

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mode = opt.mode
        in_scale, iter_num = functions.calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        opt.mode = 'train'
        dir2trained_model = functions.generate_dir2save(opt)
        if (os.path.exists(dir2trained_model)):
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = mode
        else:
            print('*** Train SinGAN for SR ***')
            real = functions.read_image(opt)
            input_h, input_w = real.shape[2], real.shape[3]
            opt.min_size = 18
            real = functions.adjust_scales2image_SR(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = mode
        print('%f' % pow(in_scale, iter_num))

        Zs_sr = []
        reals_sr = []
        NoiseAmp_sr = []
        Gs_sr = []
        real = reals[-1]  # read_image(opt)
        real_ = real
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        # for j in range(1, iter_num + 1, 1):
        #     real_ = imresize(real_, pow(1 / opt.scale_factor, 1), opt)
        #     reals_sr.append(real_)
        #     Gs_sr.append(Gs[-1])
        #     NoiseAmp_sr.append(NoiseAmp[-1])
        #     z_opt = torch.full(real_.shape, 0, device=opt.device)
        #     m = nn.ZeroPad2d(5)
        #     z_opt = m(z_opt)
        #     Zs_sr.append(z_opt)
        # out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
        # out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
        # dir2save = functions.generate_dir2save(opt)
        # plt.imsave('%s/%s_HR.png' % (dir2save,opt.input_name[:-4]), functions.convert_image_np(out.detach()), vmin=0, vmax=1)


        for j in range(iter_num):
            real_ = reals[j]
            reals_sr.append(real_)
            Gs_sr.append(Gs[j])
            NoiseAmp_sr.append(NoiseAmp[j])

            z_opt = torch.full(real_.shape, 0, device=opt.device)
            m = nn.ZeroPad2d(5)
            z_opt = m(z_opt)
            Zs_sr.append(z_opt)

        out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)

        torch.save(out, 'embeddings.pt')

        images = list(zip(*out))[0]
        embeddings = list(zip(*out))[1]

        inputs, targets = get_SR_inputs_targets(images)

        model = SR(embeddings).to('cuda:0')
        Path('sr_output').mkdir(parents=True, exist_ok=True)

        model, losses = train_SR(model, inputs, targets, num_epochs = int(opt.sr_epochs), batch_size = 64, output_dir = 'sr_output')

        base_h, base_w = images[0].shape[2], images[0].shape[3]

        print('--- generating SR predictions ---')
        predict_image(model, target_h = base_h, target_w = base_w, base_img = images[0], output_file_name = 'sr_low.png')
        predict_image(model, target_h = input_h, target_w = input_w, base_img = images[0], output_file_name = 'sr_orig.png')
        predict_image(model, target_h = input_h * 2, target_w = input_w * 2, base_img = images[0], output_file_name = 'sr_high_2x.png')
        predict_image(model, target_h = input_h * 4, target_w = input_w * 4, base_img = images[0], output_file_name = 'sr_high_4x.png')
        predict_image(model, target_h = input_h * 8, target_w = input_w * 8, base_img = images[0], output_file_name = 'sr_high_8x.png')

        plt.figure(figsize = (12, 7))
        plt.plot(range(len(losses)), losses)
        plt.savefig('loss.png')

        print('--- generating SR edges ---')
        for sr_factor in [2, 4, 8]:
            edgeSR_generate(
                f'{opt.input_dir}/{opt.input_name}', 
                sr_factor = sr_factor, 
                model = model, 
                target_h = input_h * sr_factor, 
                target_w = input_w * sr_factor, 
                base_img = images[0], 
                output_file_name = f'edge_SR_{sr_factor}x.png'
            )
        
        print('--- generating SR merges ---')
        for sr_factor in [2, 4, 8]:
            edgeSR_merge_images(f'{opt.input_dir}/{opt.input_name}', f'sr_high_{sr_factor}x.png', sr_factor)