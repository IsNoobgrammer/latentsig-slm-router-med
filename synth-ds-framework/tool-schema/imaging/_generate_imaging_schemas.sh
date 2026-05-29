#!/bin/bash
DIR="$(dirname "$0")"

# === X-RAY ===
cat > "$DIR/xray_skull.json" << 'EOF'
{
  "name": "xray_skull",
  "description": "Order a skull X-ray to evaluate bony calvarium, sella turcica, and facial bones. Used for head trauma screening, assessment of skull fractures, and shunt hardware evaluation.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["skull"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "views": {"type": "string", "enum": ["pa_lateral", "towne", "waters", "submentovertex", "caldwell"], "description": "Radiographic projection(s) requested"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., head trauma, suspected fracture, shunt evaluation)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "suspected skull fracture after blunt head trauma",
    "assessment of cranial shunt or hardware position",
    "evaluation of calvarial lytic or sclerotic lesions",
    "pre-operative planning for cranial procedures"
  ],
  "examples": [
    {
      "query": "Child fell and hit head, large scalp hematoma noted",
      "args": {"region": "skull", "modality": "xray", "views": "pa_lateral", "urgency": "urgent", "clinical_indication": "head trauma, rule out skull fracture"}
    }
  ]
}
EOF

cat > "$DIR/xray_extremity_upper.json" << 'EOF'
{
  "name": "xray_extremity_upper",
  "description": "Order an upper extremity X-ray (shoulder, humerus, elbow, forearm, wrist, or hand) to evaluate fractures, dislocations, arthritis, and foreign bodies.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["extremity-upper"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "specific_site": {"type": "string", "enum": ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand", "finger"], "description": "Specific anatomic site on upper extremity"},
      "views": {"type": "string", "enum": ["two_view", "three_view", "special_views"], "description": "Number of views requested"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., fall on outstretched hand, suspected fracture)"}
    },
    "required": ["region", "modality", "specific_site"]
  },
  "when_to_use": [
    "suspected fracture or dislocation of the upper extremity",
    "assessment of joint swelling or deformity",
    "evaluation of arthritis or degenerative changes",
    "foreign body detection in soft tissue",
    "post-reduction or post-operative follow-up"
  ],
  "examples": [
    {
      "query": "Patient fell on outstretched hand, wrist is swollen and tender over the anatomical snuffbox",
      "args": {"region": "extremity-upper", "modality": "xray", "specific_site": "wrist", "views": "three_view", "urgency": "stat", "clinical_indication": "suspected scaphoid fracture"}
    }
  ]
}
EOF

cat > "$DIR/xray_extremity_lower.json" << 'EOF'
{
  "name": "xray_extremity_lower",
  "description": "Order a lower extremity X-ray (hip, femur, knee, tibia/fibula, ankle, or foot) to evaluate fractures, dislocations, arthritis, and alignment.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["extremity-lower"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "specific_site": {"type": "string", "enum": ["hip", "femur", "knee", "tibia_fibula", "ankle", "foot", "toe"], "description": "Specific anatomic site on lower extremity"},
      "views": {"type": "string", "enum": ["two_view", "three_view", "special_views"], "description": "Number of views requested"},
      "weight_bearing": {"type": "boolean", "description": "Whether weight-bearing views are required"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., ankle sprain, knee pain, hip fracture)"}
    },
    "required": ["region", "modality", "specific_site"]
  },
  "when_to_use": [
    "suspected fracture or dislocation of the lower extremity",
    "non-weight-bearing patient with hip or knee pain after a fall",
    "assessment of joint effusion or osteoarthritis",
    "pre-operative templating for joint replacement",
    "evaluation of diabetic foot ulcers and osteomyelitis"
  ],
  "examples": [
    {
      "query": "Elderly patient fell, unable to bear weight, shortened and externally rotated leg",
      "args": {"region": "extremity-lower", "modality": "xray", "specific_site": "hip", "views": "two_view", "urgency": "stat", "clinical_indication": "suspected hip fracture"}
    }
  ]
}
EOF

cat > "$DIR/xray_spine_cervical.json" << 'EOF'
{
  "name": "xray_spine_cervical",
  "description": "Order a cervical spine X-ray to evaluate alignment, fractures, degenerative changes, and stability after trauma. Includes open-mouth odontoid view for C1-C2 assessment.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["spine-cervical"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "views": {"type": "string", "enum": ["three_view", "five_view", "flexion_extension", "swimmers"], "description": "Radiographic views requested"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., neck pain after MVC, suspected fracture)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "cervical spine trauma screening after motor vehicle collision",
    "neck pain with neurologic symptoms",
    "assessment of cervical alignment and degenerative disc disease",
    "pre-operative planning for cervical spine surgery",
    "evaluation of rheumatoid arthritis at C1-C2"
  ],
  "examples": [
    {
      "query": "Patient involved in rear-end MVC with neck stiffness and midline tenderness",
      "args": {"region": "spine-cervical", "modality": "xray", "views": "five_view", "urgency": "stat", "clinical_indication": "cervical spine trauma clearance"}
    }
  ]
}
EOF

cat > "$DIR/xray_spine_thoracic.json" << 'EOF'
{
  "name": "xray_spine_thoracic",
  "description": "Order a thoracic spine X-ray to evaluate compression fractures, metastatic disease, kyphosis, and alignment abnormalities.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["spine-thoracic"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "views": {"type": "string", "enum": ["pa_lateral", "two_view"], "description": "Radiographic views requested"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., back pain, suspected fracture, metastatic disease)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "mid-back pain with point tenderness after trauma",
    "suspected osteoporotic compression fracture",
    "known malignancy with back pain (metastatic screening)",
    "assessment of thoracic kyphosis or Scheuermann disease"
  ],
  "examples": [
    {
      "query": "Post-menopausal woman with sudden onset mid-back pain after lifting",
      "args": {"region": "spine-thoracic", "modality": "xray", "views": "two_view", "urgency": "urgent", "clinical_indication": "suspected thoracic compression fracture"}
    }
  ]
}
EOF

cat > "$DIR/xray_spine_lumbar.json" << 'EOF'
{
  "name": "xray_spine_lumbar",
  "description": "Order a lumbar spine X-ray to evaluate alignment, disc space narrowing, spondylolisthesis, fractures, and degenerative changes.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["spine-lumbar"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "views": {"type": "string", "enum": ["two_view", "four_view", "flexion_extension", "spot_lateral"], "description": "Radiographic views requested"},
      "weight_bearing": {"type": "boolean", "description": "Whether weight-bearing views are required for alignment assessment"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., low back pain, radiculopathy, trauma)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "chronic or acute low back pain initial workup",
    "suspected spondylolisthesis or spondylolysis",
    "assessment of lumbar alignment and disc degeneration",
    "post-operative fusion hardware evaluation",
    "trauma screening for lumbar burst or compression fractures"
  ],
  "examples": [
    {
      "query": "Patient with chronic low back pain and bilateral leg pain worse with extension",
      "args": {"region": "spine-lumbar", "modality": "xray", "views": "flexion_extension", "weight_bearing": true, "urgency": "routine", "clinical_indication": "suspected spondylolisthesis with instability"}
    }
  ]
}
EOF

cat > "$DIR/xray_dental.json" << 'EOF'
{
  "name": "xray_dental",
  "description": "Order a dental X-ray (periapical, bitewing, or panoramic) to evaluate tooth roots, periodontal bone, caries, and jaw pathology.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["dental"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "views": {"type": "string", "enum": ["periapical", "bitewing", "panoramic", "occlusal"], "description": "Type of dental radiograph"},
      "specific_teeth": {"type": "string", "description": "Specific teeth or quadrant to image (e.g., upper right, teeth 1-4)"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., dental pain, suspected abscess, caries detection)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "toothache or dental pain with suspected periapical pathology",
    "routine caries detection and periodontal assessment",
    "suspected dental abscess or osteomyelitis of the jaw",
    "pre-procedural assessment for dental implants or extractions",
    "evaluation of impacted wisdom teeth"
  ],
  "examples": [
    {
      "query": "Patient with severe right lower jaw pain and swelling, fever",
      "args": {"region": "dental", "modality": "xray", "views": "periapical", "specific_teeth": "lower right quadrant", "urgency": "urgent", "clinical_indication": "suspected dental abscess"}
    }
  ]
}
EOF

cat > "$DIR/xray_mammography.json" << 'EOF'
{
  "name": "xray_mammography",
  "description": "Order a mammogram for breast cancer screening or diagnostic evaluation of breast symptoms such as lump, pain, or nipple discharge.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["mammography"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "study_type": {"type": "string", "enum": ["screening", "diagnostic"], "description": "Screening for asymptomatic patients or diagnostic for symptomatic"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., screening, palpable lump, nipple discharge)"}
    },
    "required": ["region", "modality", "study_type"]
  },
  "when_to_use": [
    "routine breast cancer screening in women over 40",
    "palpable breast lump requiring diagnostic evaluation",
    "nipple discharge or skin changes on the breast",
    "follow-up of previously identified breast lesion (BI-RADS 0)",
    "high-risk screening for BRCA carriers or strong family history"
  ],
  "examples": [
    {
      "query": "45-year-old woman for routine annual screening mammogram",
      "args": {"region": "mammography", "modality": "xray", "study_type": "screening", "urgency": "routine", "clinical_indication": "routine breast cancer screening"}
    },
    {
      "query": "Patient noticed a new palpable lump in the left breast",
      "args": {"region": "mammography", "modality": "xray", "study_type": "diagnostic", "urgency": "urgent", "clinical_indication": "palpable left breast mass"}
    }
  ]
}
EOF

cat > "$DIR/xray_portable.json" << 'EOF'
{
  "name": "xray_portable",
  "description": "Order a portable (bedside) X-ray for patients who cannot be transported to the radiology department. Most commonly chest or abdomen at the bedside in the ICU or ward.",
  "category": "imaging",
  "parameters": {
    "type": "object",
    "properties": {
      "region": {"type": "string", "enum": ["chest", "abdomen"], "description": "Body region to image"},
      "modality": {"type": "string", "enum": ["xray"], "description": "Imaging modality"},
      "portable": {"type": "boolean", "description": "Confirms portable/bedside modality"},
      "urgency": {"type": "string", "enum": ["stat", "urgent", "routine"], "description": "Turnaround time needed"},
      "clinical_indication": {"type": "string", "description": "Reason for the exam (e.g., line placement check, post-intubation, respiratory status)"}
    },
    "required": ["region", "modality"]
  },
  "when_to_use": [
    "critically ill or intubated patients unable to travel to radiology",
    "verification of endotracheal tube or central line placement at bedside",
    "ICU monitoring of pulmonary status",
    "post-procedural assessment in patients on continuous monitoring",
    "trauma patients in the emergency bay"
  ],
  "examples": [
    {
      "query": "ICU patient recently intubated, check ET tube position",
      "args": {"region": "chest", "modality": "xray", "portable": true, "urgency": "stat", "clinical_indication": "post-intubation ET tube position check"}
    }
  ]
}
EOF

echo "All x-ray schemas created."