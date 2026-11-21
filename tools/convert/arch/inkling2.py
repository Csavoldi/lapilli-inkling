"""Inkling 2 Mini (GGUF architectures `inkling2` and `inkling2_loop`; Thinking Machines Lab, Apache-2.0).

The GGUF carries 24 physical layers; the looped core (layers `loop.start` .. `loop.end`) runs `loop.count`
passes over the same weights in the engine, so the pack stores the 24 layers and the loop keys go to `meta.gguf_kv`.
The experts arrive merged (`ffn_gate_up_exps`) with the gate rows first; Unsloth's mixed-precision GGUFs hold them
as NVFP4 with a per-expert FP32 `.scale` companion on some layers and as Q8_0 on the others.  The routed blobs are
requantized on V100's lanes: gate/up rows to Q4_K, down rows to Q8_0 (1152 values are not whole 256-value blocks),
Q8_0 source layers keep their bytes.  Everything else is stored as published: the Q8_0 projections are served from
`gguf`, the small tensors become F32 in `dense`.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import Arch, register

P = "inkling2_loop."

# The Apache License 2.0, with LF line endings.
APACHE2 = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

# GGUF keys of the geometry, copied to meta.geometry under the key without the prefix
GEOMETRY_KEYS = (
    "block_count", "context_length", "embedding_length", "vocab_size", "vocab_size_unpadded",
    "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.value_length",
    "attention.layer_norm_rms_epsilon", "attention.qk_clip", "attention.relative_length",
    "attention.relative_extent", "attention.relative_clip", "attention.log_scale_floor",
    "attention.log_scale_alpha", "attention.sliding_window", "attention.sliding_window_pattern",
    "attn_res.block_size", "shortconv.l_cache", "final_logit_softcapping",
    "activation.situ_beta", "activation.situ_linear_beta",
    "expert_count", "expert_used_count", "expert_feed_forward_length", "expert_weights_scale",
    "loop.start", "loop.end", "loop.count", "loop.gate_rank", "loop.varying",
    "feed_forward_length", "rope.dimension_count",
)


class Inkling2(Arch):
    name = "inkling2"
    side_sections = False
    # the BF16 small projections become F32 in `dense`; the Q8_0 projections stay in `gguf`
    form_table = {
        "attn_sink_proj.weight": "F32",
        "attn_k_conv_proj.weight": "F32",
        "attn_v_conv_proj.weight": "F32",
    }
    sampling_kv = {"general.sampling.temp": 1.0, "general.sampling.top_p": 1.0,
                   "general.sampling.top_k": 0, "general.sampling.min_p": 0.0}

    def license(self, kv):
        lic = {"name": "Apache-2.0", "text": APACHE2}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        base = kv.get("general.name") or "Inkling-2-Mini"
        stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", Path(first_gguf).name)
        tail = stem[len(base):].strip("-_ ") if stem.startswith(base) else ""
        return f"{base} {tail}" if tail else base

    def check(self, kv, own):
        from model_sections import ConvertError
        if not (kv.get("general.architecture") in ("inkling2", "inkling2_loop")):
            raise ConvertError("not an inkling2 GGUF")
        for key in ("loop.start", "loop.end", "loop.count", "loop.gate_rank", "attn_res.block_size",
                    "shortconv.l_cache", "attention.sliding_window_pattern", "activation.situ_beta",
                    "activation.situ_linear_beta", "vocab_size_unpadded"):
            if kv.get(P + key) is None:
                raise ConvertError(f"the GGUF has no {P}{key}")
        start, end, count = (int(kv[P + k]) for k in ("loop.start", "loop.end", "loop.count"))
        block = int(kv[P + "attn_res.block_size"])
        n_layer = int(kv[P + "block_count"])
        if not (0 < start < end <= n_layer and start % block == 0 and end % block == 0 and count >= 2):
            raise ConvertError(f"loop [{start}, {end}) x{count} does not fit {n_layer} layers "
                               f"or the attention-residual blocks of {block}")

    def meta_extra(self, kv, tensors, own):
        return {
            "arch": self.name,
            "geometry": {k: kv[P + k] for k in GEOMETRY_KEYS if P + k in kv},
        }


register(Inkling2(), "inkling2", "inkling2_loop")
