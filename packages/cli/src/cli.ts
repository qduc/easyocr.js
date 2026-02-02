#!/usr/bin/env node
import { Command } from 'commander';
import { version } from '@qduc/easyocr-node';

const program = new Command();

program
  .name('easyocr')
  .description('CLI for easyocr.js')
  .version(version);

program.parse();
