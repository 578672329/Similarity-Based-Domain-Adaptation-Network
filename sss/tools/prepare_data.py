import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader
from config.config import cfg

def prepare_data_CAN():

    #
    #数据集分为两部分，训练集和测试集，预处理的方式不一样
    #
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    #
    #获取源域和目标域数据集路径
    #os.path.join()路径连接函数
    #
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    #
    #只读打开类别文件，读取类别总数
    #
    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        #获得类别list
        classes = f.readlines()
        #获得去掉首位空格的类别列表(list)
        classes = [c.strip() for c in classes]

    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    #
    #准备对源域数据进行聚类的封装数据
    #
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)
    #
    #准备对目标域数据进行聚类的封装数据
    #
    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE 
    print('Building clustering_%s dataloader...' % target)
    dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # class-agnostic source dataloader for supervised training
    #
    #准备对源域数据进行训练的封装数据（有监督学习）
    #
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    # initialize the categorical dataloader
    #
    #准备带类别标签的源域和目标域数据（聚类之后的源域和目标域数据进行封装）进行训练的封装数据
    #
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type, 
                source_batch_size=source_batch_size, 
                target_batch_size=target_batch_size, 
                source_dataset_root=dataroot_S, 
                transform=train_transform, 
                classnames=classes, 
                num_workers=cfg.NUM_WORKERS,
                drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_MMD():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDatasetWithoutLabel'
    dataloaders[target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_SingleDomainSource():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

def prepare_data_SingleDomainTarget():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    target = cfg.DATASET.TARGET_NAME
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    dataloaders[target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders
