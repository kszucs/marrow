-- Copyright 2024 Szűcs Krisztián
-- SPDX-License-Identifier: Apache-2.0

-- A listing pulled in with {{< include >}} is the source file byte for byte,
-- so it opens with the licence header; the page shows the program without it.
function CodeBlock(block)
  block.text = block.text:gsub("^# Copyright .-# SPDX%-License%-Identifier: [^\n]*\n+", "", 1)
  return block
end
