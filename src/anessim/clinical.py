"""Generate coherent clinical text and structured notes from simulation events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anessim.actions import Action, ActionType
from anessim.patients import Patient
from anessim.timeline import Timeline


@dataclass
class TextEventAlignment:
    """Entry that links a span of text to one or more simulator events."""

    text_span: str
    event_indices: list[int]
    start_char: int
    end_char: int


class ClinicalTextGenerator:
    """Generate English clinical notes from a simulated case."""

    def __init__(self, rng: Any | None = None) -> None:
        # rng is accepted for future LLM/randomization usage; currently deterministic.
        self.rng = rng

    def generate(
        self,
        patient: Patient,
        timeline: Timeline,
        actions: list[Action],
        duration_min: float,
    ) -> tuple[str, list[TextEventAlignment]]:
        """Return (note, alignments)."""
        note_parts = []
        alignments: list[TextEventAlignment] = []
        char_offset = 0

        # Header
        header = self._header(patient)
        note_parts.append(header)
        char_offset += len(header) + 1

        # Preoperative assessment
        preop = self._preop_assessment(patient)
        note_parts.append(preop)
        alignments.append(
            TextEventAlignment(
                text_span=preop,
                event_indices=[],
                start_char=char_offset,
                end_char=char_offset + len(preop),
            )
        )
        char_offset += len(preop) + 1

        # Induction / intubation events
        induction_text, ind_alignment = self._induction_section(actions, char_offset)
        note_parts.append(induction_text)
        alignments.append(ind_alignment)
        char_offset += len(induction_text) + 1

        # Maintenance summary
        maint_text, maint_alignment = self._maintenance_section(actions, duration_min, char_offset)
        note_parts.append(maint_text)
        alignments.append(maint_alignment)
        char_offset += len(maint_text) + 1

        # Emergence / extubation
        emerg_text, emerg_alignment = self._emergence_section(actions, char_offset)
        note_parts.append(emerg_text)
        alignments.append(emerg_alignment)
        char_offset += len(emerg_text) + 1

        full_text = "\n\n".join(note_parts)
        return full_text, alignments

    def _header(self, patient: Patient) -> str:
        return (
            f"OPERATIVE ANESTHESIA RECORD\n"
            f"Case ID: {patient.caseid} | Age: {patient.age:.1f} y | Sex: {patient.sex} | "
            f"ASA: {patient.asa} | Procedure: {patient.opname}"
        )

    def _preop_assessment(self, patient: Patient) -> str:
        lines = [
            "PREOPERATIVE ASSESSMENT",
            f"Patient is a {patient.age:.1f}-year-old {patient.sex} with BMI {patient.bmi:.1f} kg/m2.",
            f"Planned procedure: {patient.opname} ({patient.approach}, {patient.position}).",
            f"Anesthesia plan: {patient.ane_type} anesthesia.",
        ]
        if patient.preop_htn:
            lines.append("History of hypertension.")
        if patient.preop_dm:
            lines.append("History of diabetes mellitus.")
        if patient.preop_ecg and patient.preop_ecg.lower() != "normal sinus rhythm":
            lines.append(f"Preoperative ECG: {patient.preop_ecg}.")
        else:
            lines.append("Preoperative ECG: normal sinus rhythm.")
        return "\n".join(lines)

    def _induction_section(self, actions: list[Action], char_offset: int) -> tuple[str, TextEventAlignment]:
        prop_bolus = next((a for a in actions if a.drug == "propofol" and a.action_type == ActionType.BOLUS), None)
        remi_start = next((a for a in actions if a.drug == "remifentanil" and a.action_type == ActionType.INFUSION_START), None)
        roc_bolus = next((a for a in actions if a.drug == "rocuronium"), None)
        event_indices = []
        for i, a in enumerate(actions):
            if a in (prop_bolus, remi_start, roc_bolus) and a is not None:
                event_indices.append(i)

        lines = ["INDUCTION AND AIRWAY MANAGEMENT"]
        if prop_bolus and prop_bolus.value:
            lines.append(
                f"Induction performed with propofol {prop_bolus.value:.0f} mg IV at t={prop_bolus.t_s:.0f}s."
            )
        if remi_start and remi_start.value:
            lines.append(
                f"Remifentanil infusion started at {remi_start.value:.0f} mcg/min at t={remi_start.t_s:.0f}s."
            )
        if roc_bolus and roc_bolus.value:
            lines.append(
                f"Rocuronium {roc_bolus.value:.0f} mg IV administered at t={roc_bolus.t_s:.0f}s for neuromuscular blockade."
            )
        lines.append("Airway secured via orotracheal intubation with appropriate tube size.")
        text = "\n".join(lines)
        return text, TextEventAlignment(
            text_span=text,
            event_indices=event_indices,
            start_char=char_offset,
            end_char=char_offset + len(text),
        )

    def _maintenance_section(self, actions: list[Action], duration_min: float, char_offset: int) -> tuple[str, TextEventAlignment]:
        prop_changes = [a for a in actions if a.drug == "propofol" and a.action_type == ActionType.INFUSION_CHANGE]
        sevo = [a for a in actions if a.drug == "sevoflurane"]
        vaso = [a for a in actions if a.action_type in (ActionType.VASOACTIVE_BOLUS, ActionType.VASOACTIVE_INFUSION)]
        event_indices = [i for i, a in enumerate(actions) if a in prop_changes or a in sevo or a in vaso]

        lines = ["INTRAOPERATIVE COURSE"]
        lines.append(f"Maintenance lasted approximately {duration_min:.1f} minutes.")
        if prop_changes:
            lines.append(
                f"Propofol infusion titrated {len(prop_changes)} times to maintain adequate depth of anesthesia."
            )
        if sevo:
            lines.append("Sevoflurane inhalation used as part of balanced anesthesia.")
        if vaso:
            lines.append(f"{len(vaso)} vasoactive intervention(s) were required for hemodynamic stability:")
            for a in vaso:
                lines.append(f"  - {a.drug.capitalize()} {a.value:.1f}{a.unit} at t={a.t_s:.0f}s")
        else:
            lines.append("Hemodynamics remained stable without vasoactive support.")
        lines.append("SpO2 and EtCO2 maintained within normal limits throughout the case.")
        text = "\n".join(lines)
        return text, TextEventAlignment(
            text_span=text,
            event_indices=event_indices,
            start_char=char_offset,
            end_char=char_offset + len(text),
        )

    def _emergence_section(self, actions: list[Action], char_offset: int) -> tuple[str, TextEventAlignment]:
        prop_stop = next((a for a in actions if a.drug == "propofol" and a.action_type == ActionType.INFUSION_STOP), None)
        remi_stop = next((a for a in actions if a.drug == "remifentanil" and a.action_type == ActionType.INFUSION_STOP), None)
        event_indices = []
        for i, a in enumerate(actions):
            if a in (prop_stop, remi_stop) and a is not None:
                event_indices.append(i)

        lines = ["EMERGENCE AND EXTUBATION"]
        if prop_stop:
            lines.append(f"Propofol infusion discontinued at t={prop_stop.t_s:.0f}s.")
        if remi_stop:
            lines.append(f"Remifentanil infusion discontinued at t={remi_stop.t_s:.0f}s.")
        lines.append("Patient emerged smoothly from anesthesia and was extubated uneventfully.")
        lines.append("Transferred to PACU in stable condition.")
        text = "\n".join(lines)
        return text, TextEventAlignment(
            text_span=text,
            event_indices=event_indices,
            start_char=char_offset,
            end_char=char_offset + len(text),
        )
