import CBneurons.gr as gr
import CBneurons.go as go
import CBneurons.pkj as pkj
import CBneurons.bs as bs
import CBneurons.vn as vn
import CBneurons.pons as pons

def create_neurons(subCB):
    gr.create_GR(subCB)
    go.create_GO(subCB)
    pkj.create_PKJ(subCB)
    bs.create_BS(subCB)
    vn.create_VN(subCB)
    pons.create_PONS(subCB)
