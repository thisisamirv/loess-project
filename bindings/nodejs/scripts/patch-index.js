'use strict'

const fs = require('node:fs')
const path = require('node:path')

const indexPath = path.join(__dirname, '..', 'index.js')
let index = fs.readFileSync(indexPath, 'utf8')
index = index.replace(
    /bindingPackageVersion !== '\d+\.\d+\.\d+'/g,
    "bindingPackageVersion !== require('./package.json').version"
)
index = index.replace(
    /expected \d+\.\d+\.\d+ but got/g,
    () => "expected ${require('./package.json').version} but got"
)
fs.writeFileSync(indexPath, index)

const declarationsPath = path.join(__dirname, '..', 'index.d.ts')
let declarations = fs.readFileSync(declarationsPath, 'utf8')
const asyncResultType = /(fit_async\([^\r\n]*\): Promise<)unknown(>)/
declarations = declarations.replace(asyncResultType, '$1LoessResult$2')
if (!/fit_async\([^\r\n]*\): Promise<LoessResult>/.test(declarations)) {
    throw new Error('Could not locate the generated fit_async declaration')
}
fs.writeFileSync(declarationsPath, declarations)