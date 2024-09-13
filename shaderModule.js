const renderModuleCode = /*wgsl*/`


@group(0) @binding(0) var mySampler : sampler;
@group(0) @binding(1) var myTexture : texture_2d<f32>;

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
    let pos = array(
        vec2f( 1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0, -1.0),
        vec2f( 1.0,  1.0),
        vec2f(-1.0, -1.0),
        vec2f(-1.0,  1.0),
    );

    return vec4f(pos[vertexIndex],0,1);
}


@fragment fn fragmentMain(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    let output = textureSample(myTexture, mySampler, vec2f(pixel.xy) / ${ CANVAS_WIDTH } );
    return output;
}
`

const computeModuleCode = /*wgsl*/`


struct SimData{
    dimensions: vec2f,
    frame: f32,
}

struct Ray{
    origin: vec3f,
    dir: vec3f,
    color: vec3f,
}

struct Material{
    color: vec3f,
    emmissionStrength: f32,
    smoothness: f32,
    transparency: f32,
    refractiveIndex: f32,
    materialType: f32,
}

struct Triangle{
    posA: vec3f,
    posB: vec3f,
    posC: vec3f,
    normal: vec3f,
    materialIdx: f32,
}

struct AABB{
    min: vec3f,
    startIdx: f32,
    max: vec3f,
    endIdx: f32,
    childIdx: f32,
    depth: f32,
}

struct HitInfo{
    didHit: bool,
    dist: f32,
    pos: vec3f,
    normal: vec3f,
    material: Material,
}

const groundColor = vec3f(0.6723941, 0.95839283, 1.0);
const horizonColor = vec3f(0.6523941, 0.93839283, 1.0);
const skyColor = vec3f(0.2788092, 0.56480793, 0.9264151);
const sunDir = normalize(vec3f(-0.3, 0.6, 1.0));
const sunFocus = 500.0;
const sunIntensity = 150.0;

@group(0) @binding(0) var outputTexture : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var inputTexture : texture_2d<f32>;
@group(0) @binding(2) var<uniform> camAngle: vec2f;
@group(0) @binding(3) var<uniform> camPos: vec3f;
@group(0) @binding(4) var<uniform> triangles: array<Triangle,${ NUM_TRIANGLES }>;
@group(0) @binding(5) var<uniform> boundingBoxes: array<AABB,${ NUM_BOUNDING_BOXES }>;
@group(0) @binding(6) var<uniform> materials: array<Material,${ NUM_MATERIALS }>;
@group(0) @binding(7) var<uniform> simData: SimData;


fn random(state: ptr<function, u32>) -> f32 {
    let oldState = *state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    *state = (word >> 22u) ^ word;
    return f32(*state) / 0xffffffff;
}

fn randomNormDist(state: ptr<function, u32>) -> f32 {
    var theta = 6.283185307 * random(state);
    var rho = sqrt(-2 * log(random(state)));
    return rho * cos(theta);
}

fn randomDir(state: ptr<function, u32>) -> vec3f {
    var x = randomNormDist(state);
    var y = randomNormDist(state);
    var z = randomNormDist(state);

    return normalize(vec3f(x, y, z));
}

fn rayAABBIntersect(ray: Ray, box: AABB) -> bool {
    let tbot = (box.min - ray.origin) / ray.dir;
    let ttop = (box.max - ray.origin) / ray.dir;

    let tmin = min(ttop, tbot);
    let tmax = max(ttop, tbot);
    var t = max(tmin.xx, tmin.yz);
    let t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    let t1 = min(t.x, t.y);

    return t1 >= max(t0, 0.0);
}

fn rayTriangleIntersect(ray: Ray, tri: Triangle) -> HitInfo {
    var thisHit: HitInfo;
    thisHit.didHit = false;

    let vertexA = tri.posA;
    let vertexB = tri.posB;
    let vertexC = tri.posC;
    let n = tri.normal;

    let denom = dot(n, ray.dir);
    if(denom == 0.0){
        return thisHit;
    }

    let t = dot(vertexA - ray.origin, n) / denom;
    if(t <= 0.0){
        return thisHit;
    }
    let hitPos = ray.origin + ray.dir * t;

    let edgeAB = vertexB - vertexA;
    let edgeBC = vertexC - vertexB;
    let edgeCA = vertexA - vertexC;

    let c0 = dot(cross(edgeAB, hitPos - vertexA), n);
    let c1 = dot(cross(edgeBC, hitPos - vertexB), n);
    let c2 = dot(cross(edgeCA, hitPos - vertexC), n);

    if (c0 <= 0.0 || c1 <= 0.0 || c2 <= 0.0) {
        return thisHit;
    }

    thisHit.dist = t;
    thisHit.pos = hitPos;
    thisHit.normal = n;
    thisHit.didHit = true;
    thisHit.material = materials[u32(tri.materialIdx)];
    return thisHit;
}

// Create hit info for a ray and the whole scene

fn traceRay(ray: Ray) -> HitInfo {
    var closestHit: HitInfo;
    var thisHit: HitInfo;

    var stack: array<AABB,${ BVH_MAX_DEPTH }>;
    var stackIdx = 0;
    stack[stackIdx] = boundingBoxes[0];
    stackIdx += 1;

    while(stackIdx > 0){
        stackIdx -= 1;
        let bb = stack[stackIdx];

        if(rayAABBIntersect(ray,bb)){
            if(bb.childIdx == 0){
                let start = u32(bb.startIdx);
                let end = u32(bb.endIdx);
                for(var i = start; i < end; i += 1){
                    thisHit = rayTriangleIntersect(ray, triangles[u32(i)]);
                    if((thisHit.didHit && thisHit.dist < closestHit.dist) || !closestHit.didHit){
                        closestHit = thisHit;
                    }
                }
            } else {
                stack[stackIdx] = boundingBoxes[u32(bb.childIdx)];
                stackIdx += 1;
                stack[stackIdx] = boundingBoxes[u32(bb.childIdx) + 1];
                stackIdx += 1;
            }
        }
    }

    /*rayAABBIntersect(ray, boundingBoxes[0]);
    for(var i = 0; i < ${ NUM_TRIANGLES }; i += 1){
        thisHit = rayTriangleIntersect(ray, triangles[i]);
        if((thisHit.didHit && thisHit.dist < closestHit.dist) || !closestHit.didHit){
            closestHit = thisHit;
        }
    }*/

    return closestHit;
}

fn getSky(ray: Ray) -> vec3f {
    let skyGradientT = pow(smoothstep(0.0, 0.4, ray.dir.y),0.35);
    let skyGradient = mix(horizonColor, skyColor, skyGradientT);
    let sun = pow(max(0.0, dot(ray.dir, sunDir)), sunFocus) * sunIntensity;

    let groundToSky = smoothstep(-0.01, 0.0, ray.dir.y);
    var sunMask = 0.0;
    if(groundToSky >= 1.0){
        sunMask = 1.0;
    }

    return mix(groundColor, skyGradient, groundToSky) + sun * sunMask;
}


@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) globalInvocationID: vec3u){
    let pos = globalInvocationID.xy;
    let pixel = vec2f(pos);

    let frame = simData.frame;

    let screenSize = simData.dimensions.x;
    let texCoords = vec2u(pixel);
    let pixelIndex = texCoords.x + ${ CANVAS_WIDTH } * texCoords.y;
    let planeDist = 1.0;

    var rngState = pixelIndex + u32(frame) * 719324593u;

    let rotMatX = mat3x3f(
        1.0, 0.0, 0.0,
        0.0, cos(camAngle.x), -sin(camAngle.x),
        0.0, sin(camAngle.x), cos(camAngle.x)
    );
    let rotMatY = mat3x3f(
        cos(camAngle.y), 0.0, sin(camAngle.y),
        0.0, 1.0, 0.0,
        -sin(camAngle.y), 0.0, cos(camAngle.y)
    );

    /*var planePos = vec3f(
        pixel.x / screenSize - 0.5,
        pixel.y / -screenSize + 0.5, 
        planeDist
    );*/

    let numBounces = 100;
    let numRays = 1;

    var averageLight = vec3f(0.0);

    for(var i = 0; i < numRays; i += 1){

        var planePos = vec3f(
            (pixel.x + random(&rngState) - 0.5) / screenSize - 0.5,
            (pixel.y + random(&rngState) - 0.5) / -screenSize + 0.5, 
            planeDist
        );

        var ray: Ray;
        ray.origin = camPos;
        ray.dir = normalize(rotMatY * rotMatX * planePos);
        ray.color = vec3f(1.0);

        var totalLight = vec3f(0.0);
        
        for(var j = 0; j < numBounces; j += 1){
            let traceResults = traceRay(ray);
            
            if(!traceResults.didHit){
                totalLight += getSky(ray) * ray.color;
                break;
            }

            //totalLight += getSky(ray) * traceResults.material.color; //! DELETE
            //break; //! DELETE

            let material = traceResults.material;
            let materialType = i32(material.materialType);

            switch materialType {
                case 2: {
                    let rayIsInside = select(1.0,-1.0,dot(ray.dir,traceResults.normal) > 0.0);
                    let n = traceResults.normal * rayIsInside;
                    let specularDir = reflect(ray.dir, n);
                    let diffuseDir = normalize(n + randomDir(&rngState));
                    ray.dir = mix(diffuseDir, specularDir, material.smoothness);
                    ray.origin = traceResults.pos + ray.dir * 0.001;

                    let emitted = material.color * material.emmissionStrength;
                    totalLight += emitted * ray.color;
                    ray.color *= material.color;
                }
                case 3: {
                    let indRef = material.refractiveIndex;
                    let rayIsInside = select(1.0,-1.0,dot(ray.dir,traceResults.normal) > 0.0);
                    let n = traceResults.normal * rayIsInside;

                    let cosTheta1 = dot(n,-ray.dir);
                    let cosTheta2 = sqrt(1 - (1 - cosTheta1 * cosTheta1) / (indRef * indRef));

                    let fp = (indRef * cosTheta1 - cosTheta2) / (indRef * cosTheta1 + cosTheta2);
                    let fs = (cosTheta1 - indRef * cosTheta2) / (cosTheta1 + indRef * cosTheta2);

                    var reflectance = 0.5 * (fp * fp + fs * fs);
    
                    var newDir = n + randomDir(&rngState);
                    var hitColor = material.color;
                    if(reflectance >= random(&rngState)){
                        newDir = mix(newDir, reflect(ray.dir, n), material.smoothness);
                        hitColor = vec3f(1.0);
                    }
    
                    ray.dir = normalize(newDir);
                    ray.origin = traceResults.pos + ray.dir * 0.001;
    
                    var emitted = material.color * material.emmissionStrength;
                    totalLight += emitted * ray.color;
                    ray.color *= hitColor;
                }
                case 4: {
                    let indRef = material.refractiveIndex;
                    let rayIsInside = select(1.0,-1.0,dot(ray.dir,traceResults.normal) > 0.0);
                    let indRefsRatio = select(1.0 / indRef,indRef,rayIsInside == -1.0);
                    let n = traceResults.normal * rayIsInside;

                    let cosTheta1 = dot(n,-ray.dir);
                    let cosTheta2 = sqrt(1 - (1 - cosTheta1 * cosTheta1) * indRefsRatio * indRefsRatio);

                    let fp = (cosTheta1 - cosTheta2 * indRefsRatio) / (cosTheta1 + cosTheta2 * indRefsRatio);
                    let fs = (cosTheta1 * indRefsRatio - cosTheta2) / (cosTheta1 * indRefsRatio + cosTheta2);

                    var reflectance = 0.5 * (fp * fp + fs * fs);
    
                    var newDir = refract(ray.dir,n,indRefsRatio);
                    var hitColor = vec3f(1.0);
                    if(length(newDir) < 0.1 || reflectance >= random(&rngState)){
                        newDir = reflect(ray.dir, n);
                    } else if(rayIsInside == -1.0){
                        hitColor = material.color;
                    }
    
                    ray.dir = normalize(newDir);
                    ray.origin = traceResults.pos + ray.dir * 0.0001;
    
                    var emitted = material.color * material.emmissionStrength;
                    totalLight += emitted * ray.color;
                    ray.color *= hitColor;
                }
                default {
                    let rayIsInside = select(1.0,-1.0,dot(ray.dir,traceResults.normal) > 0.0);
                    let n = traceResults.normal * rayIsInside;

                    let diffuseDir = normalize(n + randomDir(&rngState));
                    ray.dir = diffuseDir;
                    ray.origin = traceResults.pos + ray.dir * 0.001;

                    let emitted = material.color * material.emmissionStrength;
                    totalLight += emitted * ray.color;
                    ray.color *= material.color;
                }
            }

            /*let energy = (ray.color.x + ray.color.y + ray.color.z) / 3.0;
            let terminateChance = 2 * (1 - energy) / f32(numBounces);
            if(terminateChance < random(&rngState) && j > 11){
                break;
            }*/
        }
        averageLight += totalLight;
    }

    let previousColor = vec3f(textureLoad(inputTexture, pos, 0).xyz);
    averageLight = sqrt(max(averageLight / f32(numRays), vec3f(0.0)));
    let weight = 1.0/frame;
    averageLight = weight * averageLight + (1.0 - weight) * previousColor;
    textureStore(outputTexture, pos, vec4f(averageLight, 1.0));
}
`