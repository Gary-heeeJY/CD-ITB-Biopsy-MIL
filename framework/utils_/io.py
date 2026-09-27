import os
import xlwt, xlrd
import matplotlib.pyplot as plt
from xlutils.copy import copy as xl_copy

def to_cuda(tensor):
    if isinstance(tensor, (list,tuple)):
        new_tensors = []
        for i in tensor:
            new_tensors.append(i.cuda())
        return new_tensors
    else:
        return tensor.cuda()

def get_cases_name(path):
    if isinstance(path, (list, tuple)):
        cases_name = []
        for i in path:
            # cases_name.append(i.split('/')[-1].split('.')[0])
            cases_name.append(i.split('/')[-1].split('_')[1].split('_')[0])
        return cases_name
    else:
        return [path.split('/')[-1].split('.')[0]]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path, " exist!")

def metric_plot(metric_list, save_name = None, axis_x =None, ci = None):
    figg = plt.figure()
    if axis_x: # todo
        assert len(metric_list) == len(axis_x)
        plt.plot(metric_list,axis_x)
    else:
        plt.plot(metric_list)
        if ci:
            ci_lower_limit = [ci_element[0] for ci_element in ci]
            ci_upper_limit = [ci_element[1] for ci_element in ci]
            plt.plot(ci_lower_limit, linestyle="dashdot")
            plt.plot(ci_upper_limit, linestyle="dashdot")

    figg.savefig(save_name)
    plt.close()

def write_excel_xls(path, sheet_name, value, cell_overwrite_ok=False):
    index = len(value)
    if os.path.exists(path):
        workbook = xlrd.open_workbook(path)
        workbook = xl_copy(workbook)
    else:
        workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name, cell_overwrite_ok=cell_overwrite_ok)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])
    workbook.save(path)


def write_excel_xls_append(path, value, sheet_name=None):
    if os.path.exists(path):
        index = len(value)
        workbook = xlrd.open_workbook(path)
        sheets = workbook.sheet_names()
        new_workbook = xl_copy(workbook)
        if sheet_name in sheets:
            new_worksheet = new_workbook.get_sheet(sheet_name)
            rows_old = workbook.sheet_by_name(sheet_name).nrows
        else:
            new_worksheet = new_workbook.add_sheet(sheet_name)
            rows_old = 0

        for i in range(0, index):
            for j in range(0, len(value[i])):
                new_worksheet.write(i+rows_old, j, value[i][j])
        new_workbook.save(path)

    else:
        write_excel_xls(path=path, sheet_name=sheet_name, value=value)