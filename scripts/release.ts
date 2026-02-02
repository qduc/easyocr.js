import { $ } from 'bun';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';
import { spawnSync } from 'child_process';

const PACKAGES = ['core', 'node', 'web'];
const ROOT_DIR = process.cwd();

async function release() {
  const args = process.argv.slice(2);
  const isDryRun = args.includes('--dry-run');
  const targetVersion = args.find(arg => !arg.startsWith('--'));

  if (isDryRun) {
    console.log('--- DRY RUN MODE ---');
  }

  // 1. Check git status (skip if dry run)
  if (!isDryRun) {
    const status = await $`git status --porcelain`.text();
    if (status.trim() !== '') {
      console.error('❌ Working directory is not clean. Do you want to continue? (y/n)');
      if (prompt('Continue?') !== 'y') {
        process.exit(1);
      }
    }

    const branch = await $`git branch --show-current`.text();
    if (branch.trim() !== 'main' && branch.trim() !== 'master') {
      console.warn(`⚠️ You are on branch "${branch.trim()}", not main/master. Proceed? (y/n)`);
      // Standard Node/Bun prompt isn't great for interactive input in all environments,
      // but 'prompt' works in recent Bun.
      if (prompt('Proceed?') !== 'y') {
        process.exit(0);
      }
    }
  }

  // 2. Build and Test
  console.log('🚀 Building and testing all packages...');
  try {
    await $`bun run build`;
    await $`bun run test`;
  } catch (err) {
    console.error('❌ Build or tests failed. Aborting.');
    process.exit(1);
  }

  // 3. Determine new version
  const corePkgPath = join(ROOT_DIR, 'packages/core/package.json');
  const corePkg = JSON.parse(readFileSync(corePkgPath, 'utf-8'));
  const currentVersion = corePkg.version;

  let newVersion = targetVersion;
  if (!newVersion) {
    console.log(`Current version is: ${currentVersion}`);
    newVersion = prompt('Enter new version:') || '';
  }

    if (!newVersion) {
        console.error('❌ Invalid version. Aborting.');
    process.exit(1);
  }

    const isSameVersion = newVersion === currentVersion;
    if (isSameVersion) {
        console.warn(`⚠️ Version ${newVersion} is already the current version.`);
        if (prompt('Skip bumping and proceed to publish/tag? (y/n)') !== 'y') {
            process.exit(0);
        }
    }

    if (!isSameVersion) {
      console.log(`✨ Bumping versions to ${newVersion}...`);
  }

  // 4. Update all package.json files
  const updatedFiles = [];
  for (const pkgName of PACKAGES) {
    const pkgPath = join(ROOT_DIR, `packages/${pkgName}/package.json`);
    if (!existsSync(pkgPath)) continue;

    const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));
    pkg.version = newVersion;

    // Update internal dependencies
    if (pkg.dependencies) {
      for (const [dep, ver] of Object.entries(pkg.dependencies)) {
        if (dep.startsWith('@qduc/easyocr-')) {
          pkg.dependencies[dep] = `^${newVersion}`;
        }
      }
    }

      if (!isDryRun && !isSameVersion) {
      writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');
    }
    updatedFiles.push(pkgPath);
      if (!isSameVersion) {
          console.log(`   ✅ Updated @qduc/easyocr-${pkgName}`);
      }
  }

  if (isDryRun) {
    console.log('--- Dry run: skipped file writes, git operations, and publishing ---');
    console.log(`Would have published version ${newVersion} for: ${PACKAGES.join(', ')}`);
    return;
  }

  // 5. Git Commit, Tag
  console.log('💾 Committing and tagging...');
    const tagExists = (await $`git rev-parse v${newVersion} 2>/dev/null`.text().catch(() => '')).trim() !== '';

    if (tagExists) {
        console.log(`   ⏭️ Tag v${newVersion} already exists. Skipping commit and tag.`);
    } else {
        // Check if there are changes to commit
        const hasChanges = (await $`git status --porcelain`.text()).trim() !== '';
        if (hasChanges) {
            await $`git add ${updatedFiles}`;
            await $`git commit -m "release: v${newVersion}"`;
        } else {
            console.log('   ⏭️ No changes to commit.');
        }
        await $`git tag v${newVersion}`;
    }

  // 6. Publish in order
  // Order matters because of dependencies: core first, then node/web, then cli
  const publishOrder = ['core', 'node', 'web'];
  for (const pkgName of publishOrder) {
    console.log(`📦 Publishing @qduc/easyocr-${pkgName}...`);
    try {
      // Check for npm authentication
      const npmWhoami = await $`npm whoami`.text().catch(() => '');
      if (!npmWhoami.trim()) {
          spawnSync('npm', ['login'], { stdio: 'inherit' });
      }
        // Use spawnSync to ensure interactivity for OTP prompts
        const publishResult = spawnSync('npm', ['publish', '--access', 'public'], {
            cwd: join(ROOT_DIR, 'packages', pkgName),
            stdio: 'inherit'
        });
        if (publishResult.status !== 0) {
            throw new Error(`npm publish failed with exit code ${publishResult.status}`);
        }
    } catch (err) {
      console.error(`❌ Failed to publish @qduc/easyocr-${pkgName}.`);
        if (prompt('Skip this error and continue to next package? (y/n)') !== 'y') {
            process.exit(1);
        }
    }
  }

  // 7. Push
  console.log('📤 Pushing to remote...');
    spawnSync('git', ['push', 'origin', 'main', '--tags'], { stdio: 'inherit' });

  console.log(`\n🎉 Successfully released v${newVersion}!`);
}

release().catch(err => {
  console.error('💥 Unexpected error:', err);
  process.exit(1);
});
