'use strict'

const fs = require('node:fs')
const path = require('node:path')

const declarationsPath = path.join(__dirname, '..', 'index.d.ts')
let declarations = fs.readFileSync(declarationsPath, 'utf8')
const asyncResultType = /(fit_async\([^\r\n]*\): Promise<)unknown(>)/
declarations = declarations.replace(asyncResultType, '$1LoessResult$2')
if (!/fit_async\([^\r\n]*\): Promise<LoessResult>/.test(declarations)) {
    throw new Error('Could not locate the generated fit_async declaration')
}
fs.writeFileSync(declarationsPath, declarations)