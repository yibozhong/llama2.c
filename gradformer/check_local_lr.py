"""GPU checks for LR controls and the matched connected-gradient control."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.check_local import relative_error
from gradformer.local_ce import local_backward, end_to_end_backward, clip_local_gradients
from gradformer.runtime import construct, optimizer_for
from gradformer.train_local import configure_parameter_lr, validate_resume


def tiny(eta):
    torch.manual_seed(123)
    return construct('llama2', 'sgd', depth=3, width=64, heads=2, vocab=256,
                     sequence=16, step_size=eta, device='cuda')


def controls():
    x = torch.arange(32, device='cuda').reshape(2, 16)
    max_error = 0.
    for eta in (.001, .01, .1):
        for lr in (.00005, .0005, .005):
            model, reference = tiny(eta), tiny(eta)
            torch.testing.assert_close(F.softplus(model.hidden_step_raw), torch.full((3,), eta, device='cuda'))
            opt = optimizer_for(model, 'llama2', 32768); configure_parameter_lr(opt, lr)
            ref_opt = reference.configure_optimizers(.1, lr, (.9, .95), 'cuda')
            assert all(g['lr'] == g['initial_lr'] == lr for g in opt.param_groups)
            for net, optimizer in ((model, opt), (reference, ref_opt)):
                local_backward(net, x, x + 1, implementation='dense', amp=False, readout_training='final')
                clip_local_gradients(net); optimizer.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                max_error = max(max_error, (p - q).abs().max().item())
            assert all(torch.isfinite(p).all() for p in model.parameters())
    config = {k: 'unchanged' for k in ('seed','dataset','model','batch_tokens','objective','clipping','ce_implementation','cce_commit','schedule_steps','source')}
    config.update(readout_training='final', parameter_lr=.0005, training_objective='local')
    validate_resume(config, config)
    for key, value in (('parameter_lr', .005), ('model', {'hidden_step_size':.1}), ('training_objective','e2e')):
        changed = copy.deepcopy(config); changed[key] = value
        try:
            validate_resume(config, changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f'resume accepted changed {key}')
    return dict(nine_eta_lr_pairs_checked=True, configured_vs_direct_optimizer_max_error=max_error,
                resume_rejects_changed_lr_eta_objective=True)


def connected_reference():
    x = torch.arange(32, device='cuda').reshape(2, 16)
    result = {}
    for eta in (.001, .01, .1):
        for accumulation in (1, 8):
            expected, actual = tiny(eta), tiny(eta)
            errors = []
            for micro in range(accumulation):
                xx, yy = x + micro, x + micro + 1
                logits = expected(xx, yy)
                loss = F.cross_entropy(logits.float().reshape(-1, 256), yy.reshape(-1))
                (loss / (3 * accumulation)).backward()
                measured = end_to_end_backward(actual, xx, yy, accumulation=accumulation, implementation='dense', amp=False)
                torch.testing.assert_close(measured[0], loss, rtol=0, atol=0)
            for (name,p), (_,q) in zip(expected.named_parameters(), actual.named_parameters()):
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)
                errors.append((p.grad - q.grad).abs().max().item())
            result[f'eta{eta:g}_accum{accumulation}'] = max(errors)
    dense, fused = tiny(.01), tiny(.01)
    a = end_to_end_backward(dense, x, x + 1, implementation='dense')
    b = end_to_end_backward(fused, x, x + 1)
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    errors = [relative_error(q.grad, p.grad) for p,q in zip(dense.parameters(),fused.parameters())]
    assert max(errors) < .04, max(errors)
    result['bf16_cce_max_parameter_relative_l2_error'] = max(errors)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    import wandb
    run = wandb.init(project='nanochat', entity='yibozhong657-none', job_type='correctness',
                     name='gradformer-local-lr-sweep-controls',
                     config={'source':json.loads(Path('gradformer/source.json').read_text()), 'gpu':torch.cuda.get_device_name()})
    result, exit_code = {}, 1
    try:
        for name, fn in (('controls', controls), ('connected_reference', connected_reference)):
            result[name] = fn(); print(json.dumps({'check':name, 'result':result[name]}), flush=True)
        result['passed'] = True; exit_code = 0
    except Exception as error:
        result.update(passed=False, failure=f'{type(error).__name__}: {error}'); raise
    finally:
        Path(args.output).write_text(json.dumps(result,indent=2)); run.summary.update(result); run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
