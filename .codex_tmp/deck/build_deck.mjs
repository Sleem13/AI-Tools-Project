import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/Admin/Documents/GitHub/AI-Tools-Project";
const OUT = `${ROOT}/Final_Submission/Egyptian_ALPR_Final_Presentation.pptx`;
const RENDER = `${ROOT}/.codex_tmp/deck/rendered`;
const W=1280,H=720, M=48;
const C={ink:"#0B1725", navy:"#17324D", blue:"#2E74B5", cyan:"#6DCBF4", pale:"#EAF2F8", panel:"#F2F4F7", rule:"#B8BCC4", muted:"#5B6573", white:"#FFFFFF", red:"#9B1C1C"};

await fs.mkdir(RENDER,{recursive:true}); await fs.mkdir(`${ROOT}/Final_Submission`,{recursive:true});
const deck=Presentation.create({slideSize:{width:W,height:H}});

function box(slide,x,y,w,h,fill=C.panel,lineFill="none",radius=false,name=""){
  return slide.shapes.add({geometry:radius?"roundRect":"rect",name,position:{left:x,top:y,width:w,height:h},fill,line:{style:"solid",fill:lineFill,width:lineFill==="none"?0:1}});
}
function txt(slide,text,x,y,w,h,size=24,bold=false,color=C.ink,align="left",name=""){
  const s=slide.shapes.add({geometry:"textbox",name,position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=text; s.text.style={fontSize:size,bold,color,typeface:"Arial",alignment:align,verticalAlignment:"top",autoFit:"shrinkText",insets:{left:0,right:0,top:0,bottom:0}}; return s;
}
function title(slide,text,n){ txt(slide,text,M,34,1138,62,42,true,C.ink,"left","slide-title"); txt(slide,String(n).padStart(2,"0"),1185,662,45,22,14,false,C.muted,"right","slide-number"); }
function notes(slide,sourceLines=[]){ slide.speakerNotes.textFrame.setText(`[Sources]\n${sourceLines.join("\n")}`); }
function addBulletList(slide,items,x,y,w,h,size=22){ txt(slide,items.map(v=>`• ${v}`).join("\n"),x,y,w,h,size,false,C.ink); }
async function img(slide,path,x,y,w,h,fit="contain",alt=""){
  const b=await fs.readFile(path); slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:path.toLowerCase().endsWith(".png")?"image/png":"image/jpeg",alt,fit,position:{left:x,top:y,width:w,height:h}});
}

// 1 — cover, derived from Codex Grid slide 01 silhouette
{
 const s=deck.slides.add(); s.background.fill=C.white;
 txt(s,"FINAL PROJECT PRESENTATION",M,42,600,40,24,true,C.blue);
 txt(s,"Egyptian Automatic\nLicense Plate Recognition",M,178,920,220,66,true,C.ink);
 txt(s,"Dataset engineering, multi-stage YOLO detection, Arabic character decoding, and deployment workbench",M,492,780,100,27,false,C.muted);
 box(s,1005,0,275,H,C.navy); txt(s,"ALPR",1040,545,190,90,52,true,C.white,"center");
 notes(s,["Project repository and training artifacts (local workspace)."]);
}

// 2
{
 const s=deck.slides.add(); title(s,"The report moves from the problem to evidence and next steps",2);
 const items=[["01","Introduction & problem"],["02","Background & literature"],["03","Datasets & method"],["04","Metrics & results"],["05","Conclusions & future work"],["06","References"]];
 items.forEach((it,i)=>{const y=140+i*78; txt(s,it[0],M,y,75,42,25,true,C.blue); txt(s,it[1],145,y,930,42,27,false,C.ink); box(s,M,y+53,1138,1,C.rule);});
 notes(s,[]);
}

// 3
{
 const s=deck.slides.add(); title(s,"Egyptian plates turn a standard OCR task into a domain problem",3);
 txt(s,"Problem statement",M,135,420,38,25,true,C.blue);
 txt(s,"Detect and recognize Egyptian vehicle and motorcycle plates reliably from unconstrained images and video.",M,188,520,145,32,true,C.ink);
 addBulletList(s,["Arabic letters and digits with right-to-left ordering","Small targets, blur, glare, occlusion, tilt, and night scenes","Heterogeneous datasets and annotation formats","Cascaded errors can prevent downstream recognition"],M,372,540,220,22);
 await img(s,`${ROOT}/models/detection/master_plate_yolo11_20260727_132515/val_batch0_pred.jpg`,650,128,580,490,"cover","Validation examples with detected Egyptian plates");
 notes(s,["Local artifact: models/detection/master_plate_yolo11_20260727_132515/val_batch0_pred.jpg"]);
}

// 4
{
 const s=deck.slides.add(); title(s,"Prior work progressed from handcrafted segmentation to learned cascades",4);
 const cols=[
  ["Classical Egyptian ALPR","Edges, morphology, connected components, skew correction, and template matching.","2013–2014"],
  ["Deep learned localization","YOLO and CNN cascades improve robustness to background, pose, and lighting variation.","2016–2022"],
  ["Sequence & domain models","CRNN/CTC and newer Arabic-focused systems reduce reliance on perfect segmentation.","2017–2025"]];
 cols.forEach((c,i)=>{const x=M+i*397; box(s,x,175,360,355,C.panel,"none",true); txt(s,c[2],x+24,200,310,30,18,true,C.blue); txt(s,c[0],x+24,254,310,70,27,true,C.ink); txt(s,c[1],x+24,350,310,130,21,false,C.muted);});
 txt(s,"Project position: reproducible data engineering + plate detection + 38-class character detection + locale-aware decoding + web workbench",M,575,1138,58,24,true,C.navy);
 notes(s,["https://doi.org/10.1016/j.aej.2013.02.005","https://arxiv.org/abs/1506.02640","https://arxiv.org/abs/1703.07330","https://arxiv.org/abs/1507.05717","https://doi.org/10.1109/ACIRS55390.2022.9845514"]);
}

// 5
{
 const s=deck.slides.add(); title(s,"Two plate sources became one reproducible 2,317-image corpus",5);
 await img(s,`${ROOT}/reports/figures/dataset_size_comparison.png`,M,125,575,430,"contain","Dataset A and B image counts");
 const stats=[["2,551","raw images"],["2,317","harmonized"],["1,622","train"],["348","validation"],["347","test"],["6,332","character crops"]];
 stats.forEach((v,i)=>{const x=675+(i%2)*270,y=145+Math.floor(i/2)*145; box(s,x,y,240,110,C.panel,"none",true); txt(s,v[0],x+18,y+15,200,45,34,true,C.blue); txt(s,v[1],x+18,y+66,200,28,19,false,C.muted);});
 txt(s,"QA flags: near duplicates, two missing Dataset A annotations, one corrupted Dataset B image, and identity overlap in the character export.",675,585,520,64,19,false,C.red);
 notes(s,["Local reports: reports/eda/dataset_summary.md; reports/splits/split_manifest.json; reports/harmonization/harmonization_metadata.json.","Character dataset metadata: https://universe.roboflow.com/alyalsayed-vyx6g/egyptian-car-plates/dataset/13"]);
}

// 6 method diagram
{
 const s=deck.slides.add(); title(s,"The cascade narrows the search area at every stage",6);
 const xs=[55,350,645,940], y=265, w=235,h=170;
 for(let i=0;i<3;i++) s.shapes.add({geometry:"rightArrow",position:{left:xs[i]+w+18,top:y+62,width:42,height:38},fill:C.cyan,line:{style:"solid",fill:"none",width:0}});
 const stages=[
  ["1","Vehicle detection","COCO-pretrained YOLO11n","vehicle boxes"],
  ["2","Plate localization","custom YOLO11n","plate crops"],
  ["3","Character detection","38-class YOLO26n","glyph boxes"],
  ["4","Locale decoding","row clustering + RTL sort","Arabic plate text"]];
 stages.forEach((v,i)=>{box(s,xs[i],y,w,h,i===3?C.pale:C.panel,C.rule,true); txt(s,v[0],xs[i]+20,y+18,35,35,25,true,C.blue); txt(s,v[1],xs[i]+20,y+58,195,55,25,true,C.ink); txt(s,v[2],xs[i]+20,y+116,195,35,17,false,C.muted);});
 txt(s,"An optional CRNN/CTC recognizer is retained as a fallback when segmented-character weights are unavailable.",M,535,1138,58,23,false,C.navy,"center");
 notes(s,["https://docs.ultralytics.com/models/yolo11/","https://arxiv.org/abs/1507.05717","Local design: docs/two_stage_yolo.md and configs/model/two_stage.yaml."]);
}

// 7
{
 const s=deck.slides.add(); title(s,"Evaluation separates detection quality from whole-plate recognition",7);
 const metrics=[
  ["Precision","How many reported detections are correct"],["Recall","How many true objects are found"],["mAP@0.50","Average precision at IoU 0.50"],["mAP@0.50:0.95","Localization quality across stricter IoUs"],["CER","Character-level edit error"],["Exact match","Perfect plate strings / all plates"]];
 metrics.forEach((m,i)=>{const x=M+(i%2)*590,y=135+Math.floor(i/2)*150; txt(s,m[0],x,y,250,40,25,true,C.blue); txt(s,m[1],x,y+47,500,55,21,false,C.ink); box(s,x,y+113,520,1,C.rule);});
 txt(s,"Current status",M,593,180,28,18,true,C.red); txt(s,"Detector metrics are available; CER, exact-match accuracy, end-to-end recall, latency, and FPS remain to be benchmarked.",235,588,950,46,21,true,C.ink);
 notes(s,["https://docs.ultralytics.com/guides/yolo-performance-metrics/"]);
}

// 8
{
 const s=deck.slides.add(); title(s,"Plate localization is strong, and the 50-epoch run is the best checkpoint",8);
 await img(s,`${ROOT}/models/detection/master_plate_yolo11_20260727_132515/results.png`,M,125,700,500,"contain","YOLO11 plate detector training curves");
 const vals=[["96.24%","mAP@0.50"],["82.38%","mAP@0.50:0.95"],["92.02%","precision"],["91.24%","recall"]];
 vals.forEach((v,i)=>{const x=805+(i%2)*205,y=160+Math.floor(i/2)*160; box(s,x,y,185,125,C.panel,"none",true); txt(s,v[0],x+15,y+22,155,42,30,true,C.blue); txt(s,v[1],x+15,y+76,155,36,17,false,C.muted);});
 txt(s,"Best recorded at epoch 49. The longer 100-epoch run did not improve the stricter mAP metric.",805,520,405,88,21,true,C.navy);
 notes(s,["Local metrics: models/detection/master_plate_yolo11_20260727_060023/results.csv and models/detection/master_plate_yolo11_20260727_132515/results.csv."]);
}

// 9
{
 const s=deck.slides.add(); title(s,"Character detection finds symbols well but localizes them less precisely",9);
 const vals=[["97.93%","mAP@0.50"],["63.75%","mAP@0.50:0.95"],["94.90%","precision"],["94.37%","recall"]];
 vals.forEach((v,i)=>{const x=M+(i%2)*255,y=150+Math.floor(i/2)*165; box(s,x,y,225,130,C.panel,"none",true); txt(s,v[0],x+18,y+24,190,43,32,true,C.blue); txt(s,v[1],x+18,y+82,190,28,18,false,C.muted);});
 await img(s,`${ROOT}/models/character/yolo26_characters/labels.jpg`,600,120,600,470,"contain","Character class and bounding-box distributions");
 txt(s,"Best recorded at epoch 27 across 38 digit and Arabic-letter classes.",M,545,500,70,22,true,C.navy);
 txt(s,"Interpretation: characters are usually detected, but tight small-object boxes remain harder at stricter IoU thresholds.",600,610,600,52,19,false,C.red);
 notes(s,["Local metrics: models/character/yolo26_characters/results.csv.","Local artifact: models/character/yolo26_characters/labels.jpg."]);
}

// 10
{
 const s=deck.slides.add(); title(s,"The present evidence supports component readiness—not a final ALPR claim",10);
 const blocks=[
  ["What is working","Reproducible data pipeline; plate detector; 38-class character detector; right-to-left decoder; API and React workbench."],
  ["What the curves suggest","Plate performance stabilizes early. Longer training trades recall for a small precision increase without better strict mAP."],
  ["What limits interpretation","Near duplicates and nine shared base identities across character splits can inflate validation performance."],
  ["What is still missing","Leakage-controlled end-to-end exact match, CER, latency, FPS, and video-level robustness."]];
 blocks.forEach((b,i)=>{const x=M+(i%2)*590,y=140+Math.floor(i/2)*245; box(s,x,y,540,205,i===3?C.pale:C.panel,"none",true); txt(s,b[0],x+24,y+25,485,38,25,true,i===3?C.red:C.blue); txt(s,b[1],x+24,y+82,485,100,21,false,C.ink);});
 notes(s,["Local README.md and training artifacts."]);
}

// 11
{
 const s=deck.slides.add(); title(s,"A leakage-controlled benchmark is the next milestone",11);
 const steps=[
  ["Now","Freeze grouped splits","Group by source, vehicle, and original plate identity."],
  ["Next","Benchmark the cascade","Report exact match, CER, recall, FP/image, latency, and FPS."],
  ["Then","Improve hard cases","Per-class balancing, tilt/night/motorcycle analysis, threshold calibration."],
  ["Deploy","Optimize and learn","Tracking + multi-frame voting, edge export, active learning from reviews."]];
 box(s,72,348,1135,2,C.rule);
 steps.forEach((v,i)=>{const x=72+i*282; box(s,x,339,18,18,C.blue,"none",true); txt(s,v[0],x,270,190,28,18,true,C.blue); txt(s,v[1],x,390,235,54,24,true,C.ink); txt(s,v[2],x,465,235,100,19,false,C.muted);});
 txt(s,"Conclusion",M,120,180,32,19,true,C.blue); txt(s,"The project has a solid, reproducible foundation and strong component metrics. Final value depends on proving full-string recognition under an unbiased test protocol.",M,165,1100,78,30,true,C.navy);
 notes(s,["Local project conclusions based on repository metrics and QA reports."]);
}

// 12
{
 const s=deck.slides.add(); title(s,"References",12);
 const refs=[
  "[1] El-Adawi et al., Automated New License Plate Recognition in Egypt, 2013.",
  "[2] Abd El Rahman et al., Automatic Arabic Number Plate Recognition, 2013.",
  "[3] Redmon et al., You Only Look Once, CVPR 2016.",
  "[4] Masood et al., License Plate Detection and Recognition Using Deeply Learned CNNs, 2017.",
  "[5] Shi et al., CRNN for Image-Based Sequence Recognition, TPAMI 2017.",
  "[6] Youssef et al., A New Benchmark Dataset for Egyptian LPR, ACIRS 2022.",
  "[7] Youssef et al., Real-time Egyptian LPR using YOLO, IJACSA 2022.",
  "[8] Abdellatif et al., Low-Cost IoT-Based Arabic LPR, ASEJ 2023.",
  "[9] Ultralytics, YOLO11 and performance-metrics documentation.",
  "[10] EALPR dataset repository and Roboflow Egyptian character dataset."
 ];
 txt(s,refs.slice(0,5).join("\n\n"),M,125,550,490,19,false,C.ink); txt(s,refs.slice(5).join("\n\n"),655,125,560,490,19,false,C.ink);
 notes(s,["https://doi.org/10.1016/j.aej.2013.02.005","https://arxiv.org/abs/1506.02640","https://arxiv.org/abs/1703.07330","https://arxiv.org/abs/1507.05717","https://doi.org/10.1109/ACIRS55390.2022.9845514","https://docs.ultralytics.com/models/yolo11/","https://docs.ultralytics.com/guides/yolo-performance-metrics/","https://github.com/ahmedramadan96/EALPR"]);
}

for (const [i,s] of deck.slides.items.entries()) {
  const blob=await deck.export({slide:s,format:"png",scale:1});
  await fs.writeFile(`${RENDER}/slide-${String(i+1).padStart(2,"0")}.png`,Buffer.from(await blob.arrayBuffer()));
  const layout=await s.export({format:"layout"}); await fs.writeFile(`${RENDER}/slide-${String(i+1).padStart(2,"0")}.layout.json`,await layout.text());
}
const montage=await deck.export({format:"webp",montage:true,scale:1}); await fs.writeFile(`${RENDER}/montage.webp`,Buffer.from(await montage.arrayBuffer()));
const pptx=await PresentationFile.exportPptx(deck); await pptx.save(OUT);
console.log(OUT);
