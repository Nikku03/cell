# 5-organism cascade test (3-stream and 4-stream)

Real labels per organism. Streams: conservation, phase2 (MOCK from conservation), cooccur, flux (where FBA model exists).

## beril_Keio  (4-stream)

n=3,585, true essentials=605 (16.9%)

### per-stream
| stream | coverage | precision | recall | n_called |
|---|---:|---:|---:|---:|
| conservation | 0.992 | 0.608 | 0.795 | 791 |
| phase2 | 0.992 | 0.597 | 0.775 | 786 |
| cooccur | 0.917 | 0.424 | 0.603 | 860 |
| flux | 0.000 | 0.000 | 0.000 | 0 |

### tiered cascade
| tier | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| >= 1/4 agree | 0.422 | 0.841 | 0.337 | 1207 |
| >= 2/4 agree | 0.607 | 0.793 | 0.221 | 791 |
| >= 3/4 agree | 0.743 | 0.539 | 0.122 | 439 |
| >= 4/4 agree | 0.000 | 0.000 | 0.000 | 0 |
| UNANIMOUS (gold) | 0.712 | 0.550 | 0.131 | 468 |

## mtub  (4-stream)

n=3,638, true essentials=760 (20.9%)

### per-stream
| stream | coverage | precision | recall | n_called |
|---|---:|---:|---:|---:|
| conservation | 0.000 | 0.000 | 0.000 | 0 |
| phase2 | 0.000 | 0.000 | 0.000 | 0 |
| cooccur | 0.000 | 0.000 | 0.000 | 0 |
| flux | 0.272 | 0.690 | 0.237 | 261 |

### tiered cascade
| tier | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| >= 1/4 agree | 0.000 | 0.000 | 0.000 | 0 |
| >= 2/4 agree | 0.000 | 0.000 | 0.000 | 0 |
| >= 3/4 agree | 0.000 | 0.000 | 0.000 | 0 |
| >= 4/4 agree | 0.000 | 0.000 | 0.000 | 0 |
| UNANIMOUS (gold) | 0.000 | 0.000 | 0.000 | 0 |

## beril_Putida  (4-stream)

n=4,715, true essentials=917 (19.4%)

### per-stream
| stream | coverage | precision | recall | n_called |
|---|---:|---:|---:|---:|
| conservation | 0.887 | 0.679 | 0.675 | 911 |
| phase2 | 0.887 | 0.674 | 0.651 | 886 |
| cooccur | 0.971 | 0.412 | 0.541 | 1203 |
| flux | 0.299 | 0.542 | 0.154 | 260 |

### tiered cascade
| tier | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| >= 1/4 agree | 0.440 | 0.774 | 0.342 | 1614 |
| >= 2/4 agree | 0.657 | 0.674 | 0.199 | 940 |
| >= 3/4 agree | 0.720 | 0.485 | 0.131 | 618 |
| >= 4/4 agree | 0.909 | 0.087 | 0.019 | 88 |
| UNANIMOUS (gold) | 0.684 | 0.324 | 0.092 | 434 |

## beril_BFirm  (3-stream)

n=5,983, true essentials=2560 (42.8%)

### per-stream
| stream | coverage | precision | recall | n_called |
|---|---:|---:|---:|---:|
| conservation | 0.890 | 0.838 | 0.366 | 1118 |
| phase2 | 0.890 | 0.844 | 0.356 | 1081 |
| cooccur | 0.870 | 0.758 | 0.420 | 1417 |

### tiered cascade
| tier | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| >= 1/3 agree | 0.743 | 0.532 | 0.306 | 1833 |
| >= 2/3 agree | 0.849 | 0.367 | 0.185 | 1107 |
| >= 3/3 agree | 0.919 | 0.243 | 0.113 | 676 |
| UNANIMOUS (gold) | 0.911 | 0.243 | 0.114 | 684 |

## beril_Burk376  (3-stream)

n=5,549, true essentials=2140 (38.6%)

### per-stream
| stream | coverage | precision | recall | n_called |
|---|---:|---:|---:|---:|
| conservation | 0.884 | 0.866 | 0.417 | 1031 |
| phase2 | 0.884 | 0.865 | 0.408 | 1010 |
| cooccur | 0.867 | 0.721 | 0.457 | 1356 |

### tiered cascade
| tier | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| >= 1/3 agree | 0.715 | 0.574 | 0.310 | 1720 |
| >= 2/3 agree | 0.872 | 0.422 | 0.187 | 1035 |
| >= 3/3 agree | 0.955 | 0.286 | 0.116 | 642 |
| UNANIMOUS (gold) | 0.953 | 0.287 | 0.116 | 645 |

## Macro across the 5 organisms

Note: 3-stream macro is over all 5; 4-stream macro over the 3 FBA-equipped orgs only.
