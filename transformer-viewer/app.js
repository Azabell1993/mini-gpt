function byId(id){return document.getElementById(id)}
const seq = byId('seq'), dmodel = byId('dmodel'), layers = byId('layers'), heads = byId('heads')
const seqVal = byId('seq_val'), dVal = byId('d_val'), nVal = byId('n_val'), hVal = byId('h_val')
const shape1 = byId('shape1'), mhTxt = byId('mh_txt'), layersTxt = byId('layers_txt')
function update(){
  seqVal.textContent = seq.value
  dVal.textContent = dmodel.value
  nVal.textContent = layers.value
  hVal.textContent = heads.value
  shape1.textContent = `(${seq.value} × ${dmodel.value})`
  mhTxt.textContent = `h=${heads.value}, d_h=d/h=${Math.floor(dmodel.value/heads.value)}`
  layersTxt.textContent = `N=${layers.value}`
}
;['input','change'].forEach(ev=>{
  seq.addEventListener(ev, update)
  dmodel.addEventListener(ev, update)
  layers.addEventListener(ev, update)
  heads.addEventListener(ev, update)
})
update()
