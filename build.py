import os


def build_directories(run_folder):

    if not os.path.exists(run_folder):

        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, 'storage'))
        os.mkdir(os.path.join(run_folder, 'output'))
        os.mkdir(os.path.join(run_folder, 'weights'))
        os.mkdir(os.path.join(run_folder, 'visual_library'))

        print('[i] Directories Set...\n')

    return None


    
