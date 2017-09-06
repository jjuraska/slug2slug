#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import codecs
import json
from argparse import ArgumentParser
from tempfile import mkdtemp
import os
import shutil
import subprocess
import re
import sys

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def read_lines(file_name, multi_ref=False):
    """Read one instance per line from a text file. In multi-ref mode, assumes multiple lines
    (references) per instance & instances separated by empty lines."""
    buf = [[]] if multi_ref else []
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            line = line.strip()
            if multi_ref:
                if not line:
                    buf.append([])
                else:
                    buf[-1].append(line)
            else:
                buf.append(line)
    if multi_ref and not buf[-1]:
        del buf[-1]
    return buf


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


def create_mteval_file(refs, path, file_type):
    """Given references/outputs, create a MTEval .sgm XML file.
    @param refs: data to store in the file (human references/system outputs/dummy sources)
    @param path: target path where the file will be stored
    @param file_type: the indicated "set type" (ref/tst/src)
    """
    # swap axes of multi-ref data (to 1st: different refs, 2nd: instances) & pad empty references
    data = [[]]
    for inst_no, inst in enumerate(refs):
        if not isinstance(inst, list):  # single-ref data
            inst = [inst]
        for ref_no, ref in enumerate(inst):
            if len(data) <= ref_no:  # there's more refs than previously known: pad with empty
                data.append([''] * inst_no)
            data[ref_no].append(ref)
        ref_no += 1
        while ref_no < len(data):  # less references than previously: pad with empty
            data[ref_no].append('')
            ref_no += 1

    with codecs.open(path, 'wb', 'UTF-8') as fh:
        settype = file_type + 'set'
        fh.write('<%s setid="%s" srclang="any" trglang="%s">\n' % (settype, 'e2e', 'en'))
        for inst_set_no, inst_set in enumerate(data):
            sysid = file_type + ('' if len(data) == 1 else '_%d' % inst_set_no)
            fh.write('<doc docid="test" genre="news" origlang="any" sysid="%s">\n<p>\n' % sysid)
            for inst_no, inst in enumerate(inst_set, start=1):
                fh.write('<seg id="%d">%s</seg>\n' % (inst_no, inst))
            fh.write('</p>\n</doc>\n')
        fh.write('</%s>' % settype)


def evaluate(ref_file, sys_file):
    """Main procedure, running the MS-COCO & MTEval evaluators on the given files."""

    # read input files
    data_ref = read_lines(ref_file, multi_ref=True)
    data_sys = read_lines(sys_file)
    # dummy source files (have no effect on measures, but MTEval wants them)
    data_src = [''] * len(data_sys)

    # create temp directory
    temp_path = mkdtemp(prefix='e2e-eval-')
    print >> sys.stderr, 'Creating temp directory ', temp_path

    scores = {}
	
#    # convert references to MS-COCO format in-memory
#    coco_ref = create_coco_refs(data_ref)
#    # create COCO test file (in a temporary file)
#    coco_sys = create_coco_sys(data_sys)
#    coco_sys_file = os.path.join(temp_path, 'coco_sys.json')
#    with open(coco_sys_file, 'wb') as coco_sys_fh:
#        json.dump(coco_sys, coco_sys_fh)
#
#    # run the MS-COCO evaluator
#    print >> sys.stderr, 'Running MS-COCO evaluator...'
#    coco = COCO()
#    coco.dataset = coco_ref
#    coco.createIndex()
#
#    coco_res = coco.loadRes(coco_sys_file)
#    coco_eval = COCOEvalCap(coco, coco_res)
#    coco_eval.evaluate()
#    scores = {metric: score for metric, score in coco_eval.eval.items()}

    # create MTEval files
    mteval_ref_file = os.path.join(temp_path, 'mteval_ref.sgm')
    create_mteval_file(data_ref, mteval_ref_file, 'ref')
    mteval_sys_file = os.path.join(temp_path, 'mteval_sys.sgm')
    create_mteval_file(data_sys, mteval_sys_file, 'tst')
    mteval_src_file = os.path.join(temp_path, 'mteval_src.sgm')
    create_mteval_file(data_src, mteval_src_file, 'src')
    mteval_log_file = os.path.join(temp_path, 'mteval_log.txt')

    # run MTEval
    print >> sys.stderr, 'Running MTEval to compute BLEU & NIST...'
    mteval_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'mteval', 'mteval-v13a-sig.pl')
    mteval_out = subprocess.check_output(['perlbrew', 'exec',
                                          'perl', mteval_path,
                                          '-r', mteval_ref_file,
                                          '-s', mteval_src_file,
                                          '-t', mteval_sys_file,
                                          '-f', mteval_log_file], stderr=subprocess.STDOUT)
    scores['NIST'] = float(re.search(r'NIST score = ([0-9.]+)', mteval_out).group(1))
    scores['BLEU'] = float(re.search(r'BLEU score = ([0-9.]+)', mteval_out).group(1))
    print >> sys.stderr, mteval_out

    # print out the results
    print 'SCORES:\n=============='
    for metric in ['BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr']:
        print '%s: %.4f' % (metric, scores[metric])
    print

    # delete the temporary directory
    print >> sys.stderr, 'Removing temp directory'
    shutil.rmtree(temp_path)


if __name__ == '__main__':
    ap = ArgumentParser(description='E2E Challenge evaluation -- MS-COCO & MTEval wrapper')
    ap.add_argument('ref_file', type=str, help='references file -- multiple references separated by empty lines')
    ap.add_argument('sys_file', type=str, help='system output file to evaluate')
    args = ap.parse_args()

    evaluate(args.ref_file, args.sys_file)
