import argparse
import datetime
import os
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import util
import madrys
import models
import matplotlib.pyplot as plt
from evaluator import Evaluator
from trainer import Trainer
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
mlconfig.register(madrys.MadrysLoss)
plt.rcParams["figure.figsize"] = (30,15)
# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--save_frequency', default=-1, type=int)
# Datasets Options
parser.add_argument('--train_face', action='store_true', default=False)
parser.add_argument('--train_portion', default=1.0, type=float)
parser.add_argument('--train_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_path', type=str, default='../datasets')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--poison_rate', default=1.0, type=float)
parser.add_argument('--perturb_tensor_filepath', default=None, type=str)
args = parser.parse_args()


# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    train_loss = []
    train_acc = []
    eval_loss = []
    eval_acc = []
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        #print('epoch for training loss',epoch,trainer.loss_meters.avg)
        train_loss.append(trainer.loss_meters.avg)
        train_acc.append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            ENV['curren_acc'] = evaluator.acc_meters.avg*100
            #print("epoch for eval loss",epoch,evaluator.loss_meters.avg)
            eval_loss.append(evaluator.loss_meters.avg)
            eval_acc.append(evaluator.acc_meters.avg*100)
            ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
        else:
            pass
            # model.eval()
            # model.module.classify = True
            # evaluator.eval(epoch, model)
            # payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            # logger.info(payload)
            # model.classify = False
            # identity_list = lfw_test.get_lfw_list('lfw_test_pair.txt')
            # img_paths = [os.path.join('../datasets/lfw-112x112', each) for each in identity_list]
            # eval_acc = lfw_test.lfw_test(model, img_paths, identity_list, 'lfw_test_pair.txt', args.eval_batch_size, logger=logger)
            # ENV['curren_acc'] = eval_acc
            # ENV['best_acc'] = max(ENV['best_acc'], eval_acc)
            # ENV['eval_history'].append(eval_acc)
            # # Reset Stats
            # trainer._reset_stats()
            # evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        is_best=is_best,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)

        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            filename = checkpoint_path_file + '_epoch%d' % (epoch)
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=filename)
            logger.info('Model Saved at %s', filename)
    figure, axis = plt.subplots(2,2)
  
    # For Train Loss
    axis[0, 0].plot(train_loss)
    axis[0, 0].set_title("Train Loss")
    axis[0,0].set_xlabel("Epochs",fontweight='bold')
    axis[0,0].set_ylabel("Loss",fontweight='bold')

     # For Eval Loss
    axis[0, 1].plot(eval_loss)
    axis[0, 1].set_title("Evaluation Loss")
    axis[0,1].set_xlabel("Epochs",fontweight='bold')
    axis[0,1].set_ylabel("Loss",fontweight='bold')

    # For Train Acc
    axis[1,0].plot(train_acc)
    axis[1,0].set_title("Training Accuracy")
    axis[1,0].set_xlabel("Epoch",fontweight='bold')
    axis[1,0].set_ylabel("Accuracy",fontweight='bold')
    # For Eval Acc
    axis[1,1].plot(eval_acc)
    axis[1,1].set_title("Evaluation Accuracy")
    axis[1,1].set_xlabel("Epoch",fontweight='bold')
    axis[1,1].set_ylabel("Accuracy",fontweight='bold')

    plt.show()
    plt.savefig("losses_and_acc_imagenet_min_entropy_GN.png")

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in (data_loader['test_dataset']):
        #print("I",i)
        #print("sample",sample)
        output = model(inputs.cuda()) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes


   #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #random 
 #   classes = ['lacewing', 'Shetland sheepdog','bee','groom','leafhopper','poncho', 'Irish wolfhound','beer glass','capuchin','window shade','puck','hornbill','colobus', 'dial telephone','chest','ambulance','lab coat','snail','bloodhound','night snake','plate rack','Welsh springer spaniel','ibex','Loafer','African chameleon','buckeye','barrow','sunscreen','cornet','refrigerator','hammer','sax','bolo tie','king snake','shoe shop','sports car',
#'piggy bank','triceratops','warplane','hair spray','toyshop','bathtub','jellyfish','boxer','photocopier','Chihuahua','tricycle','black-footed ferret','crash helmet','space bar','espresso','pitcher','crate','sorrel','typewriter keyboard','sandal','tailed frog', 'Australian terrier','vestment','lighter','briard','baboon','cricket','quail','thatch','European gallinule','tripod','Bouvier des Flandres','cloak','triumphal arch','overskirt','hay','patas','holster','Lakeland terrier','kelpie','German short-haired pointer','remote control','bullet train','lens cap','Saluki','doormat','odometer',' moving van','black-and-tan coonhound', 'Dandie Dinmont','necklace','comic book','soft-coated wheaten terrier','long-horned beetle','Tibetan terrier','meerkat','nail','jigsaw puzzle','brassiere','ear','milk can','sturgeon','puffer',' great grey owl']



    #min 
    classes = ["hatchet","lacewing","tick","barn spider","web site","parachute","backpack","lampshade","combination lock","ruler","assault rifle","radio",
		"puck","chambered nautilus","gasmask","power drill","cassette","white stork","spotlight","iron"," bald eagle","red wine","cornet","hammer","carpenter's kit",
		"sax"," bolo tie","letter opener","maraca","hook","bulletproof vest","soap dispenser","jack-o'-lantern","mailbag","kite","muzzle","mouse","warplane","hand-held computer","scabbard",
		"ballpoint","cleaver","pick","digital watch","jellyfish","matchstick","microphone","printer","bee eater","nematode","spoonbill","nipple","vine snake","red-backed sandpiper",
		"syringe","oil filter","electric guitar","buckle","analog clock","stopwatch","joystick","lighter","quill","pencil sharpener","switch","cocktail shaker","can opener",
		"cassette player","screwdriver","perfume","loupe","remote control","knee pad","mousetrap","fountain pen","screw","espresso maker","corkscrew","stage","redshank","dowitcher",
		"safety pin","modem"," loudspeaker","scale","magnetic compass","binder","necklace","face powder","reflex camera","hourglass","whistle","hair slide","abaya","envelope","digital clock",
	"hand blower","candle","theater curtain","red-breasted merganser"]
#max classes = ["guinea pig","maypole","Shetland sheepdog","zucchini","lesser panda","recreational vehicle","police van","minivan","convertible","pot","pizza","tobacco shop","ringlet","garter snake","giant panda","koala","golfcart","minibus","Pekinese","ambulance","jinrikisha","ox","bolete","paddlewheel" ,"standard schnauzer","Blenheim spaniel","mobile home","barbershop","cheeseburger","patio","rain barrel","shoe shop","potpie","garbage truck","bull mastiff","beach wagon"," fire engine","Leonberg","cliff dwelling","carbonara","toyshop","hot pot","cardoon","sorrel","Tibetan mastiff","pickup","bagel","guacamole","Australian terrier","Dungeness crab","library","dock","butcher shop","king crab","electric locomotive","trifle","mashed potato","gyromitra","cucumber","echidna","jackfruit","orangutan","bookshop","hog","horse cart","box turtle","Model T","clumber","hen-of-the-woods","trolleybus","gorilla","streetcar","greenhouse","bullet train","coral reef","carousel","acorn squash","lumbermill","admiral","hamster","four-poster","brain coral","confectionery","school bus","hotdog","oxcart","jaguar","gondola","Border terrier","silky terrier","monarch","monastery","Tibetan terrier","grocery store","burrito","jeep","meat loaf","bakery","restaurant","apiary" ]
 # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                    columns = [i for i in classes])
    plt.figure(figsize = (120,70))
    plt.xlabel('Truth label')
    plt.ylabel('Predicted')
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output_imagenet_min_entropy_GN.png')

    return






def main():
    model = config.model().to(device)
    datasets_generator = config.dataset(train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        poison_rate=args.poison_rate,
                                        perturb_type=args.perturb_type,
                                        patch_location=args.patch_location,
                                        perturb_tensor_filepath=args.perturb_tensor_filepath,
                                        seed=args.seed)
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))
    if 'Poison' in args.train_data_type:
        with open(os.path.join(exp_path, 'poison_targets.npy'), 'wb') as f:
            if not (isinstance(datasets_generator.datasets['train_dataset'], dataset.MixUp) or isinstance(datasets_generator.datasets['train_dataset'], dataset.CutMix)):
                poison_targets = np.array(datasets_generator.datasets['train_dataset'].poison_samples_idx)
                np.save(f, poison_targets)
                logger.info(poison_targets)
                logger.info('Poisoned: %d/%d' % (len(poison_targets), len(datasets_generator.datasets['train_dataset'])))
                logger.info('Poisoned samples idx saved at %s' % (os.path.join(exp_path, 'poison_targets')))
                logger.info('Poisoned Class %s' % (str(datasets_generator.datasets['train_dataset'].poison_class)))

    if args.train_portion == 1.0:
        print("train portion full")
        data_loader = datasets_generator.getDataLoader()
        train_target = 'train_dataset'
    else:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                               train_shuffle=True,
                                                               train_drop_last=True)

    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': [],
           'cm_history': []}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    if args.train:
        train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
