# Synthetic Vietnamese clinical notes

These three notes were **written for this demo**. They describe fictional patients, contain no real patient data, and are
not medical advice.

| File | Content |
|---|---|
| `note_01.txt` | Outpatient endocrinology: hypertension, newly diagnosed type 2 diabetes, HbA1c / glucose |
| `note_02.txt` | Respiratory admission: community-acquired pneumonia, CRP / WBC, chest X-ray |
| `note_03.txt` | Nephrology follow-up: CKD stage 3 as a complication of type 2 diabetes, losartan (ARB) |
| `note_0N.extraction.json` | Hand-made reference extraction in the `Extraction` schema of `docs/AGENT_SPEC.md` §5 |

The extractions were validated on 2026-10-11:
- every relation is in the closed vocabulary;
- every `h`/`t` refers to an entity of the same file;
- every `evidence_vi` is a substring of its note (case-insensitive).

Uses:
- the mock LLM replays them, so ingestion costs $0;
- tests use them as fixtures;
- a live extraction can be compared against them (entity/triple F1).

Shared concepts link the notes after commit (keys `med:{type}:{slug(name_en)}`): `hypertension` (cases 1, 2, 3),
`type 2 diabetes` (1, 3), `amlodipine` (1, 2), `dyspnea` (1, 2), `elevated` (all).

## Demo questions (medical graph)

| Question (VI) | Expected answer | Shape |
|---|---|---|
| Bệnh nhân nào đang dùng amlodipine? | Patient (case 1), Patient (case 2) | 1 hop, inverse |
| Viêm phổi cộng đồng được điều trị bằng thuốc gì? | amoxicillin/clavulanic acid | 1 hop |
| Losartan thuộc nhóm thuốc nào? | angiotensin II receptor blocker | 1 hop |
| Biến chứng của đái tháo đường type 2 được điều trị bằng thuốc gì? | losartan | **2 hops via a bridge** (type 2 diabetes → CKD stage 3 → losartan): the AER case |
| Xét nghiệm nào của ca 3 cho kết quả giảm? | eGFR | 2 hops through a LabResult node (CVT-like) |
| Bệnh nhân ca 1 có triệu chứng gì, và không có triệu chứng gì? | headache, dizziness; denies chest pain, dyspnea | 1 hop, assertion-aware |

These are functional checks for the demo (M6), not a benchmark. WL was trained on WebQSP (Freebase), so medical retrieval
is zero-shot.
